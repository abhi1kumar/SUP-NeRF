import json
import os
import numpy as np
import cv2
import imageio
import tqdm
import torch
import pytorch3d.transforms.rotation_conversions as rot_trans

from src.data_nuscenes import NuScenesData
from model_supnerf import SUPNeRF
from src.utils import roi_process, preprocess_img_square, render_box, normalize_by_roi, \
    view_points, view_points_batch, corners_of_box, corners_of_box_batch, Resize, roi_coord_trans, \
    image_float_to_uint8, get_random_pose2, get_rays, sample_from_rays_v2, ray_box_intersection, \
    render_rays_v2 
from src.renderer import volume_rendering3, render_rays_v3


class OptimizerDemo:
    def __init__(self, gpu, hpams, model_file=None, reg_iters=3, save_dir='demo_output', ray_batch_size=2048,
                 rend_aabb=True, adjust_scale=1.0):
        super().__init__()
        self.hpams = hpams
        self.use_bn = True
        if 'norm_layer_type' in self.hpams['net_hyperparams'].keys() and \
                self.hpams['net_hyperparams']['norm_layer_type'] != 'BatchNorm2d':
            self.use_bn = False
        self.reg_iters = reg_iters
        self.save_dir = save_dir
        self.rand_angle_lim = 0  # limits to generate random 3 rotation angles besides yaw
        self.ray_batch_size = ray_batch_size
        self.rend_aabb = rend_aabb
        self.adjust_scale = adjust_scale
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model(model_file)

        # shapecode, texturecode, poses
        self.shapecodes = []
        self.texturecodes = []
        self.obj_poses = []
        self.obj_wlh = []

    def make_model(self):
        self.model = SUPNeRF(**self.hpams['net_hyperparams']).to(self.device)

    def load_model(self, model_file=None):
        saved_data = torch.load(model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

        # mean shape should only consider those optimized codes when some of those are not touched
        if 'optimized_idx' in saved_data.keys():
            optimized_idx = saved_data['optimized_idx'].numpy()
            self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'][optimized_idx > 0],
                                         dim=0).reshape(1, -1)
            self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'][optimized_idx > 0],
                                           dim=0).reshape(1, -1)
        else:
            self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1, -1)
            self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1, -1)

    def set_optimizers(self, shapecode, texturecode):
        self.update_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': self.hpams['optimize']['lr_shape']},
            {'params': texturecode, 'lr': self.hpams['optimize']['lr_texture']}
        ])

    def set_optimizers_w_poses(self, shapecode, texturecode, rots, trans):
        self.update_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': self.hpams['optimize']['lr_shape']},
            {'params': texturecode, 'lr': self.hpams['optimize']['lr_texture']},
            {'params': rots, 'lr': self.hpams['optimize']['lr_pose']},
            {'params': trans, 'lr': self.hpams['optimize']['lr_pose']}
        ])

    def update_learning_rate(self):
        opt_values = self.nopts // self.hpams['optimize']['lr_half_interval']
        self.hpams['optimize']['lr_shape'] = self.hpams['optimize']['lr_shape'] * 2 ** (-opt_values)
        self.hpams['optimize']['lr_texture'] = self.hpams['optimize']['lr_texture'] * 2 ** (-opt_values)
        self.hpams['optimize']['lr_pose'] = self.hpams['optimize']['lr_pose'] * 2 ** (-opt_values)

    def save_img3(self, gen_rgb_imgs, gen_depth_imgs, gt_imgs, obj_id, n_opts,
                  psnr=None, depth_err=None, rot_err=None, trans_err=None, cross_name=None):
        # H, W = gt_imgs[0].shape[:2]
        W_tgt = np.min([gt_img.shape[1] for gt_img in gt_imgs])

        gen_rgb_imgs = torch.cat(gen_rgb_imgs).reshape(-1, W_tgt, 3)
        gen_depth_imgs = torch.cat(gen_depth_imgs).reshape(-1, W_tgt)
        gt_imgs = torch.cat(gt_imgs).reshape(-1, W_tgt, 3)
        H_cat = gen_rgb_imgs.shape[0]

        # compute a normalized version to visualize
        gen_depth_imgs = (gen_depth_imgs - torch.mean(gen_depth_imgs)) / torch.std(gen_depth_imgs)
        gen_depth_imgs = gen_depth_imgs - torch.min(gen_depth_imgs)
        gen_depth_imgs = gen_depth_imgs / (torch.max(gen_depth_imgs) - torch.min(gen_depth_imgs))
        gen_depth_imgs = gen_depth_imgs.reshape(-1, W_tgt, 1).repeat(1, 1, 3)

        ret = torch.zeros(H_cat, 3 * W_tgt, 3)
        ret[:, :W_tgt, :] = gen_rgb_imgs
        ret[:, W_tgt:2*W_tgt, :] = gen_depth_imgs
        ret[:, 2*W_tgt:3*W_tgt, :] = gt_imgs
        ret = image_float_to_uint8(ret.detach().cpu().numpy())

        # plot scores:
        if psnr is not None and depth_err is not None:
            err_str1 = 'PSNR: {:.3f},  Depth err: {:.3f}'.format(
                psnr[0], depth_err[0])
            ret = cv2.putText(ret, err_str1, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0))
        if rot_err is not None and trans_err is not None:
            err_str2 = 'R err: {:.3f},  T err: {:.3f}'.format(
                rot_err[0], trans_err[0])
            ret = cv2.putText(ret, err_str2, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0))

        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if cross_name is None:
            imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(n_opts) + '.png'), ret)
        else:
            imageio.imwrite(os.path.join(save_img_dir, cross_name + '.png'), ret)

    def output_single_view_vis(self, vis_img, mask_occ, cam_pose, obj_sz, K, roi, shapecode, texturecode,
                               wlh, log_idx, psnr=None, depth_err=None, errs_R=None, errs_T=None, vis_im_sz=128):
        with torch.no_grad():
            if not self.rend_aabb:
                obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
                rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(self.model, self.device,
                                                                        vis_img, mask_occ, cam_pose,
                                                                        obj_diag, K,
                                                                        roi, self.hpams['n_samples'],
                                                                        shapecode, texturecode,
                                                                        self.hpams['shapenet_obj_cood'],
                                                                        self.hpams['sym_aug'],
                                                                        im_sz=vis_im_sz, n_rays=None)
            else:
                rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v3(self.model, self.device,
                                                                        vis_img, mask_occ, cam_pose,
                                                                        obj_sz, K,
                                                                        roi, self.hpams['n_samples'],
                                                                        shapecode, texturecode,
                                                                        self.hpams['shapenet_obj_cood'],
                                                                        self.hpams['sym_aug'],
                                                                        im_sz=vis_im_sz, n_rays=None,
                                                                        adjust_scale=self.adjust_scale)
            # TODO: add maks based on depth_rays to remove background artifacts
            generated_img = rgb_rays.view((vis_im_sz, vis_im_sz, 3))
            generated_depth = depth_rays.view((vis_im_sz, vis_im_sz))
            # gt_img = tgt_img.clone()[:, roi[1]:roi[3], roi[0]:roi[2], :].cpu()
            vis_img = vis_img.unsqueeze(0).permute((0, 3, 1, 2))
            vis_img = Resize((vis_im_sz, vis_im_sz))(vis_img).permute((0, 2, 3, 1))[0]
            gt_occ = occ_pixels.view((vis_im_sz, vis_im_sz)).cpu()

            # draw 3D box given predicted pose
            est_R = cam_pose[:3, :3].numpy().T
            est_T = -est_R @ cam_pose[:3, 3:].numpy()
            pred_uv = view_points(
                corners_of_box(np.concatenate([est_R, est_T], axis=1), wlh),
                K.numpy(), normalize=True)
            pred_uv[0, :] -= roi[0].item()
            pred_uv[1, :] -= roi[1].item()
            g_c = np.array([0, 1, 0]).astype(np.float64)

            u_vec_new, v_vec_new = roi_coord_trans(pred_uv[0, :], pred_uv[1, :],
                                                   roi.numpy(), im_sz_tgt=vis_im_sz)
            pred_uv[0, :] = u_vec_new
            pred_uv[1, :] = v_vec_new
            # generated_img = render_box(generated_img.cpu().numpy(), pred_uv, colors=(g_c, g_c, g_c))
            vis_img = render_box(vis_img.cpu().numpy(), pred_uv, colors=(g_c, g_c, g_c))

            self.save_img3([generated_img],
                           [generated_depth],
                           [torch.from_numpy(vis_img)], log_idx, self.nopts,
                           psnr, [depth_err], errs_R, errs_T)

    def fw_pose_update(self,
                       im_feat_batch,
                       src_pose_batch,
                       wlh_batch,
                       roi_batch,
                       K_batch,
                       K_inv_batch,
                       iters=3,
                       ):
        pose_per_iter = []
        pose_per_iter.append(src_pose_batch.squeeze().cpu().numpy())
        for iter in range(0, iters):
            src_pose_batch = self.fw_pose_one_step(
                im_feat_batch,
                src_pose_batch,
                wlh_batch,
                roi_batch,
                K_batch,
                K_inv_batch)
            pose_per_iter.append(src_pose_batch.squeeze().cpu().numpy())

        return pose_per_iter

    def fw_pose_one_step(self,
                         im_feat_batch,
                         src_pose_batch,
                         wlh_batch,
                         roi_batch,
                         K_batch,
                         K_inv_batch,
                         ):

        src_uv_batch = view_points_batch(corners_of_box_batch(src_pose_batch, wlh_batch), K_batch, normalize=True)

        # normalize src_uv_batch to align with img_in_batch frame, now normalized to (-1, 1) x (-1, 1)
        src_uv_norm, dim_batch = normalize_by_roi(src_uv_batch[:, :2, :], roi_batch, need_square=True)

        # regress delta_pose may be better than delta uv which may not consistent which is more dependent on ransac
        bsize = im_feat_batch.shape[0]
        delta_pose_batch = self.model.pose_update(im_feat_batch, src_uv_norm.view((bsize, -1)))

        # un-normalize delta_pose_batch to expected scope. Network output is assumed to around (-1, 1)
        # delta_pose_batch = torch.sigmoid(delta_pose_batch)
        delta_pose_batch[:, :3] *= (torch.pi * 2)
        delta_pose_batch[:, 3:5] *= dim_batch.unsqueeze(-1)
        delta_pose_batch[:, 5] += 1

        # apply delta_pose to original pose representation
        rot_vec_src = rot_trans.matrix_to_axis_angle(src_pose_batch[:, :, :3])
        pred_rot_vec = rot_vec_src + delta_pose_batch[:, :3]
        pred_R = rot_trans.axis_angle_to_matrix(pred_rot_vec)

        T_src = src_pose_batch[:, :, 3:]
        src_pose_uv = torch.matmul(K_batch, T_src)
        pred_u = src_pose_uv[:, 0] / src_pose_uv[:, 2] + delta_pose_batch[:, 3:4]
        pred_v = src_pose_uv[:, 1] / src_pose_uv[:, 2] + delta_pose_batch[:, 4:5]
        pred_Z = src_pose_batch[:, 2, 3:] * delta_pose_batch[:, 5:]
        pred_T = torch.cat([pred_u * pred_Z, pred_v * pred_Z, pred_Z], dim=1).unsqueeze(-1)
        pred_T = torch.matmul(K_inv_batch, pred_T)

        pred_pose_batch = torch.cat([pred_R, pred_T], dim=2)

        return pred_pose_batch

    def optimize_pose_nerf(self, objects_data):
        """
            For a list of detected objects in an image, neural reconstruct each object's shape, texuture and estimate
            each object's 6D pose in image coordinate frame
        """

        full_img = objects_data['img']
        rois = objects_data['rois']
        masks_occ = objects_data['masks_occ']
        K_batch = objects_data['cam_intrinsics'].unsqueeze(0)
        K_inv_batch = torch.linalg.inv(K_batch)

        H, W = full_img.shape[0:2]
        # Per object
        for obj_idx, roi in enumerate(rois):
            # t0 = time.time()
            print(f'optimizing object: {obj_idx+1}/{len(rois)}')

            # roi process may differ for dl_update input image and nerf render setups
            roi_new1 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=False)

            # crop tgt img to roi
            tgt_img = full_img[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]]
            full_occ = masks_occ[obj_idx].clone()
            mask_occ = full_occ[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)

            # preprocess image and
            img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
            # t1 = time.time()

            # predict initial codes and object dim
            shapecode, texturecode, posecode, _, wlh_batch = self.model.encode_img(img_in.to(self.device))

            wlh_batch = wlh_batch.detach().cpu()
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            # Use the mean to improve completeness
            shapecode = (shapecode + self.mean_shape.clone().to(self.device)) / 2
            texturecode = (texturecode + self.mean_texture.clone().to(self.device)) / 2
            shapecode = shapecode.detach().requires_grad_()
            texturecode = texturecode.detach().requires_grad_()

            # t2 = time.time()
            # print(f'image coding time {t2-t1}s')

            # initialize with fully random pose (blind guess depth = 20, and fully random yaw)
            init_obj_pose = get_random_pose2(K_batch[0].numpy(),
                                             roi.numpy(),
                                             yaw_lim=np.pi, angle_lim=self.rand_angle_lim,
                                             trans_lim=0.3, depth_fix=20)

            #  run pose refiner before nerf optimization (TEMP not include in the overall iterations)
            with torch.no_grad():
                pose_per_iter = self.fw_pose_update(
                    posecode,
                    torch.from_numpy(init_obj_pose.astype(np.float32)).unsqueeze(0).to(self.device),
                    wlh_batch.to(self.device),
                    roi_new1.unsqueeze(0).to(self.device),
                    K_batch.to(self.device),
                    K_inv_batch.to(self.device),
                    iters=self.reg_iters)
            # t3 = time.time()
            # print(f'pose estimation time {t3-t2}s')
            pose_per_iter = torch.from_numpy(np.array(pose_per_iter)).to(self.device)
            pred_pose = pose_per_iter[-1]

            # set pose parameters
            rot_mat_vec = pred_pose[:3, :3].unsqueeze(0)
            trans_vec = pred_pose[:3, 3].unsqueeze(0).to(self.device).detach().requires_grad_()
            rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimization
            self.nopts = 0
            self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            while self.nopts < self.hpams['optimize']['num_opts']:
                self.opts.zero_grad()
                loss_per_img = []

                # the first a few to load pre-computed poses for evaluation
                if self.nopts > self.reg_iters:
                    t2opt = trans_vec[0].unsqueeze(-1)
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])
                else:
                    t2opt = pose_per_iter[self.nopts, :3, 3:]
                    rot_mat2opt = pose_per_iter[self.nopts, :3, :3]

                # convert object pose to camera pose
                rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
                t2opt = -rot_mat2opt @ t2opt
                cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # A different roi and crop for nerf
                roi_new2 = roi_process(roi, H, W, roi_margin=5, sq_pad=True)

                # crop tgt img to roi
                tgt_img = full_img[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]]
                vis_img = tgt_img.clone()
                mask_occ = full_occ[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ <= 0)

                # render ray values and prepare target rays
                if not self.rend_aabb:
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                               tgt_img, mask_occ, cam2opt,
                                                                                               obj_diag, K_batch[0],
                                                                                               roi_new2,
                                                                                               self.hpams['n_samples'],
                                                                                               shapecode, texturecode,
                                                                                               self.hpams[
                                                                                                   'shapenet_obj_cood'],
                                                                                               self.hpams['sym_aug'],
                                                                                               im_sz=64,
                                                                                               n_rays=None
                                                                                               )
                else:
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v3(self.model, self.device,
                                                                                               tgt_img, mask_occ, cam2opt,
                                                                                               obj_sz, K_batch[0],
                                                                                               roi_new2,
                                                                                               self.hpams['n_samples'],
                                                                                               shapecode, texturecode,
                                                                                               self.hpams[
                                                                                                   'shapenet_obj_cood'],
                                                                                               self.hpams['sym_aug'],
                                                                                               im_sz=64,
                                                                                               n_rays=None,
                                                                                               adjust_scale=self.adjust_scale
                                                                                               )

                # Compute losses
                # Critical to let rgb supervised on white background
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                        torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # Occupancy loss
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                   torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                loss.backward()

                # save the rgb loss for psnr computation (only include object mask, random sampled for efficiency)
                mask_rgb = occ_pixels.clone()
                mask_rgb[occ_pixels < 0] = 0
                loss_rgb2 = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb) + 1e-9)
                loss_per_img.append(loss_rgb2.detach().item())

                # save pose loss
                pred_obj_R = cam2opt[:, :3].detach().cpu().transpose(-2, -1)
                pred_obj_T = -pred_obj_R @ (cam2opt[:, 3:].detach().cpu())
                pred_obj_pose = torch.cat([pred_obj_R, pred_obj_T], dim=-1).unsqueeze(0)

                # The first a few is just for evaluation purpose to render rays, do not update codes or pose
                if self.nopts > self.reg_iters:
                    self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                if (self.nopts <= self.reg_iters or self.nopts == (self.hpams['optimize']['num_opts'] - 1)):
                    self.output_single_view_vis(vis_img, mask_occ, cam2opt.detach().cpu(), obj_sz, K_batch[0],
                                                roi_new2, shapecode, texturecode, obj_sz, f'obj_{obj_idx}')
                self.nopts += 1

            # t4 = time.time()
            # print(f'NeRF time {t4 - t3}s')
            # print(f'overall time per sample {t4 - t0}s')

            # Save the optimized codes
            self.shapecodes.append(shapecode.detach().cpu().clone())
            self.texturecodes.append(texturecode.detach().cpu().clone())
            self.obj_poses.append(pred_obj_pose.detach().cpu().clone())
            self.obj_wlh.append(wlh_batch.detach().cpu().clone())

        self.shapecodes = torch.cat(self.shapecodes, dim=0)
        self.texturecodes = torch.cat(self.texturecodes, dim=0)
        self.obj_poses = torch.cat(self.obj_poses, dim=0)
        self.obj_wlh = torch.cat(self.obj_wlh, dim=0)

    def vis_scene(self, objects_data, manipulation):
        """
            Render a list of objects from reconstructed shape and texture with poses in a new image
            obj_poses: (N, 3, 4)
            obj_whl: (N, 3)
            shape_codes: (N, D)
            texture_codes: (N, D)
            K: (3, 3)
            manipulation: temporarily [dx, dy, dz] in cam coordinates
        """
        # full_img = objects_data['img']
        # rois_org = objects_data['rois']
        K = objects_data['cam_intrinsics']

        Nb = len(self.obj_poses)
        H = self.hpams['dataset']['img_h']
        W = self.hpams['dataset']['img_w']
        n_samples = self.hpams['n_samples']
        all_rays = torch.ones((H, W, Nb, 8), dtype=torch.float32) * (-1)

        K_batch = K.unsqueeze(0).repeat(Nb, 1, 1)
        # manipulate objects to new camera frame
        obj_poses = self.obj_poses.clone()
        obj_poses[:, :, 3] += torch.tensor(manipulation, dtype=torch.float32).unsqueeze(0)

        # compute the targe region based on manipulated objects
        corners_3d_batch = corners_of_box_batch(obj_poses, self.obj_wlh, is_kitti=False)  # (Nb, 3, 8)
        corners_2d_batch = view_points_batch(corners_3d_batch, K_batch, normalize=True)  # (Nb, 3, 8)

        # calculate each object's roi
        rois = torch.zeros((Nb, 4), dtype=torch.float32)  # (Nb, 4)
        rois[:, 0] = corners_2d_batch[:, 0].min(axis=1)[0]
        rois[:, 1] = corners_2d_batch[:, 1].min(axis=1)[0]
        rois[:, 2] = corners_2d_batch[:, 0].max(axis=1)[0]
        rois[:, 3] = corners_2d_batch[:, 1].max(axis=1)[0]
        rois = rois.type(torch.int32)
        for ii, roi in enumerate(rois):
            rois[ii] = roi_process(roi, H, W, roi_margin=0, sq_pad=False)
        # rois[:, 0] = torch.maximum(rois[:, 0], torch.as_tensor(0))
        # rois[:, 1] = torch.maximum(rois[:, 1], torch.as_tensor(0))
        # rois[:, 2] = torch.minimum(rois[:, 2], torch.as_tensor(W-1))
        # rois[:, 3] = torch.minimum(rois[:, 3], torch.as_tensor(H-1))

        # prepare rays for N objects covering the union of area of all objects.
        # For occluding objects a single ray can have different in and out values
        obj_diag_list = []
        for obj_i, roi in enumerate(rois):
            obj_pose = obj_poses[obj_i]
            R_c2o = obj_pose[:3, :3].transpose(0, 1)
            t_c2o = - R_c2o @ np.expand_dims(obj_pose[:3, 3], -1)
            cam_pose = torch.cat([R_c2o, t_c2o], dim=1)

            # ray center and dir for each pixel in object frame
            rays_o, viewdir = get_rays(K, cam_pose, roi)

            obj_diag = np.linalg.norm(self.obj_wlh[obj_i]).astype(np.float32)
            obj_diag_list.append(obj_diag)
            xmin, ymin, xmax, ymax = roi

            all_rays[ymin:ymax, xmin:xmax, obj_i, :3] = rays_o.view(ymax-ymin, xmax-xmin, -1) / (obj_diag/2)
            all_rays[ymin:ymax, xmin:xmax, obj_i, 3:6] = viewdir.view(ymax-ymin, xmax-xmin, -1)

            if self.rend_aabb:
                # get samples along each ray
                obj_w, obj_l, obj_h = self.obj_wlh[obj_i]
                aabb_min = np.asarray([-obj_l / obj_diag, -obj_w / obj_diag, -obj_h / obj_diag]).reshape((1, 3)).repeat(
                    rays_o.shape[0], axis=0)
                aabb_max = np.asarray([obj_l / obj_diag, obj_w / obj_diag, obj_h / obj_diag]).reshape((1, 3)).repeat(
                    rays_o.shape[0], axis=0)
                # aabb_min = None
                # aabb_max = None
                # ATTENTION: assign accurate aabb is important for multi-object rendering
                z_in, z_out, intersect = ray_box_intersection(rays_o.cpu().detach().numpy() / (obj_diag/2),
                                                              viewdir.cpu().detach().numpy(),
                                                              aabb_min=aabb_min,
                                                              aabb_max=aabb_max)

                bound_near = all_rays[ymin:ymax, xmin:xmax, obj_i, 6].flatten(0, 1)
                bound_far = all_rays[ymin:ymax, xmin:xmax, obj_i, 7].flatten(0, 1)
                bound_near[intersect] = torch.from_numpy(z_in)
                bound_far[intersect] = torch.from_numpy(z_out)
                all_rays[ymin:ymax, xmin:xmax, obj_i, 6] = bound_near.view(ymax-ymin, xmax-xmin)
                all_rays[ymin:ymax, xmin:xmax, obj_i, 7] = bound_far.view(ymax-ymin, xmax-xmin)
            else:
                z_in = torch.linalg.norm(cam_pose[:, -1]) - obj_diag/2
                z_out = torch.linalg.norm(cam_pose[:, -1]) + obj_diag/2
                all_rays[ymin:ymax, xmin:xmax, obj_i, 6] = z_in / (obj_diag/2)
                all_rays[ymin:ymax, xmin:xmax, obj_i, 7] = z_out / (obj_diag/2)

        obj_diag_list = torch.tensor(obj_diag_list, dtype=torch.float32)
        far_bounds = all_rays[:, :, :, 7].view(H*W, Nb)
        near_bounds = all_rays[:, :, :, 6].view(H*W, Nb)
        valid_indices = (far_bounds - near_bounds).max(-1)[0] > 0
        valid_rays = all_rays.view(H*W, -1, 8)[valid_indices, ...]

        with torch.no_grad():
            rgb_map = list()
            for batch_rays in tqdm.tqdm(torch.split(valid_rays, self.ray_batch_size)):
                rays = batch_rays.view(-1, 8)  # (Nr * Nb, 8)
                Nr = batch_rays.shape[0]

                z_coarse = sample_from_rays_v2(rays, n_samples)
                empty_space = z_coarse == -1

                xyz = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
                viewdir = rays[:, 3:6]
                viewdir = viewdir.unsqueeze(-2).repeat(1, n_samples, 1)

                # compute the distance values in physical scale
                obj_diag_list_new_view = obj_diag_list.view(1, -1, 1, 1).repeat(Nr, 1, 1, 1).flatten(0, 1)
                z_vals = torch.norm((xyz - rays[:, None, :3]) * (obj_diag_list_new_view / 2), p=2, dim=-1)
                z_vals[empty_space] = -1
                # z_vals = z_coarse * (obj_diag_list_new_view.squeeze(-1) / 2)

                # inference samples on the ray
                xyz = xyz.view(Nr, Nb, n_samples, 3).permute((1, 0, 2, 3)).flatten(0, 1)
                viewdir = viewdir.view(Nr, Nb, n_samples, 3).permute((1, 0, 2, 3)).flatten(0, 1)
                # temporal code due to conflict of training scale of normalized coordinates
                xyz *= self.adjust_scale
                if self.hpams['shapenet_obj_cood']:
                    xyz = xyz[:, :, [1, 0, 2]]
                    xyz[:, :, 0] *= (-1)
                    viewdir = viewdir[:, :, [1, 0, 2]]
                    viewdir[:, :, 0] *= (-1)
                # our model accept (Nb x ray_batch_size, n_samples, 3) for xyz and viewdir, need permutation before
                sigmas, rgbs = self.model(xyz.to(self.device),
                                          viewdir.to(self.device),
                                          self.shapecodes.to(self.device),
                                          self.texturecodes.to(self.device))
                rgbs = rgbs.view(Nb, Nr, n_samples, 3).permute((1, 0, 2, 3)).flatten(0, 1)
                sigmas = sigmas.view(Nb, Nr, n_samples).permute((1, 0, 2)).flatten(0, 1)
                # empty space assigned to white
                rgbs[empty_space, ...] = 1
                sigmas[empty_space] = 0

                # sort by z values
                z_vals = z_vals.view(-1, Nb*n_samples)
                z_sort = torch.sort(z_vals, 1).values
                z_args = torch.searchsorted(z_sort, z_vals)
                rgbs = rgbs.view(-1, Nb*n_samples, 3).cpu()
                rgbs_sort = torch.zeros_like(rgbs).scatter_(1, z_args[:, :, None].repeat(1, 1, 3), rgbs)
                sigmas = sigmas.view(-1, Nb*n_samples).cpu()
                sigmas_sort = torch.zeros_like(sigmas).scatter_(1, z_args, sigmas)

                rgb, depth, weights = volume_rendering3(sigmas_sort, rgbs_sort, z_sort, white_bkgd=True)

                rgb_map.append(rgb)

            rgb_map = torch.cat(rgb_map, 0)

        canvas = torch.ones(H * W, 3).type_as(all_rays)
        canvas[valid_indices, :] = rgb_map
        canvas = (canvas.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)

        return canvas


if __name__ == '__main__':
    tgt_img_name = 'n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984240912467.jpg'
    gpu = 0
    model_file = 'checkpoints/supnerf/epoch_39.pth'
    ray_batch_size = 1024
    save_dir = os.path.join('demo_output', tgt_img_name[:-4])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Read Hyper-parameters
    with open('jsonfiles/hpam_demo.json', 'r') as f:
        hpams = json.load(f)

    # load data
    nusc_data_dir = hpams['dataset']['test_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_instance')
    nusc_version = hpams['dataset']['test_nusc_version']
    det3d_path = os.path.join(nusc_data_dir, 'pred_det3d')

    # TODO: rule out those truncated boxes
    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='val',
        debug=False,
        add_pose_err=2,
        det3d_path=det3d_path,
        pred_box2d=False,
        selfsup=True
    )

    # initialize NeRF optimizer
    optimizer = OptimizerDemo(gpu, hpams, model_file=model_file, save_dir=save_dir,
                              ray_batch_size=ray_batch_size, rend_aabb=True, adjust_scale=1.0)

    # get predicted objects and masks associated with each image
    objects_data = nusc_dataset.get_objects_in_image(tgt_img_name)

    # TODO: visualize masks, predicted 3D boxes
    imageio.imwrite(os.path.join(save_dir, tgt_img_name), (objects_data['img'].numpy()*255).astype(np.uint8))

    if objects_data is None:
        exit(0)
    # optimize on objects from an image
    optimizer.optimize_pose_nerf(objects_data)

    trans_vec = [[0, 0, 0],
                 [-1, 0, 1],
                 [-2, 0, 2],
                 [-3, 0, 3],
                 [-4, 0, 4],
                 [-5, 0, 5]]
    # render with object manipulation
    print('Novel-view Rendering frame by frame ...')
    with imageio.get_writer(os.path.join(save_dir, 'scene.gif'), mode='I', duration=0.5) as writer:
        for delta_t in trans_vec:
            canvas = optimizer.vis_scene(objects_data, delta_t)
            # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            writer.append_data(canvas)
    writer.close()
    exit(0)
