import os
import imageio
import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch3d.transforms.rotation_conversions as rot_trans
from torchvision.transforms import Resize

from utils import image_float_to_uint8, align_imgs_width, render_rays, render_rays_v2, \
    render_rays_specified, render_full_img, render_virtual_imgs, view_points_batch, corners_of_box_batch, \
    calc_pose_err, preprocess_img_square, preprocess_img_keepratio, normalize_by_roi, render_box, \
    view_points, corners_of_box, obj_pose_kitti2nusc, obj_pose_nuse2kitti, roi_coord_trans, roi_process
from skimage.metrics import structural_similarity as compute_ssim
from model_autorf import AutoRF, AutoRFMix
from model_codenerf import CodeNeRF
from model_supnerf import SUPNeRF

# CODE_SAVE_ITERS_ = [0, 2, 4, 8, 16, 32]
CODE_SAVE_ITERS_ = [0, 5, 10, 20, 50, 100]
BOX_FAC = 1.1

WLH_MEAN = np.array([1.9446588, 4.641784, 1.7103361])
WLH_STD = np.array([0.1611075, 0.3961748, 0.20885137])

box_c = np.array([1, 144 / 255, 30 / 255]).astype(np.float64)
vis_im_sz = 128
puttext_ratio = vis_im_sz/128


class OptimizerWaymo:
    def __init__(self, gpu, waymo_dataset, hpams, hpams_pose_refiner=None, hpams_pose_regressor=None, model_epoch=None,
                 opt_pose=False, ada_posecode=0, reg_iters=3, opt_multiview=False, pred_wlh=False,
                 save_postfix='_waymo', num_workers=0, shuffle=False, save_freq=100, vis=0,):
        super().__init__()
        self.hpams = hpams
        self.hpams_pose_refiner = hpams_pose_refiner
        self.hpams_pose_regressor = hpams_pose_regressor
        self.use_bn = 'norm_layer_type' in self.hpams['net_hyperparams'].keys() and \
                      self.hpams['net_hyperparams']['norm_layer_type'] == 'BatchNorm2d'
        self.model_epoch = model_epoch
        self.opt_pose = opt_pose
        self.ada_posecode = ada_posecode
        self.reg_iters = reg_iters
        self.opt_multiview = opt_multiview
        self.pred_wlh = pred_wlh
        self.save_freq = save_freq
        self.vis = vis
        self.save_postfix = save_postfix
        self.nusc2waymo_boxfac = BOX_FAC
        if 'pred_wlh' in hpams['net_hyperparams'].keys() and hpams['net_hyperparams']['pred_wlh'] > 0 and self.pred_wlh:
            self.nusc2waymo_boxfac = 1.0  # if to use pred_wlh, no need to adjust ratio for the refiner
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model(self.model_epoch)
        self.make_save_img_dir()
        print('we are going to save at ', self.save_dir)
        self.waymo_dataset = waymo_dataset
        self.dataloader = DataLoader(self.waymo_dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

        # initialize shapecode, texturecode, poses to optimize
        self.optimized_shapecodes = {}
        self.optimized_texturecodes = {}
        self.optimized_poses = {}

        # optimized poses are used for cross-view evaluation
        # poses are saved per img (per time step, per camera)
        latent_dim = self.hpams['net_hyperparams']['latent_dim']
        n_code = len(CODE_SAVE_ITERS_)
        for ii, sample in enumerate(self.waymo_dataset.all_valid_samples):
            (data_idx, obj_idx) = sample
            if data_idx not in self.optimized_poses.keys():
                self.optimized_poses[data_idx] = {}
            if obj_idx not in self.optimized_poses[data_idx].keys():
                self.optimized_poses[data_idx][obj_idx] = torch.zeros((n_code, 3, 4), dtype=torch.float32)

        # codes are saved per img (per time step, per camera)
        for ii, sample in enumerate(self.waymo_dataset.all_valid_samples):
            (data_idx, obj_idx) = sample
            if data_idx not in self.optimized_shapecodes.keys():
                self.optimized_shapecodes[data_idx] = {}
                self.optimized_texturecodes[data_idx] = {}
            if obj_idx not in self.optimized_shapecodes[data_idx].keys():
                self.optimized_shapecodes[data_idx][obj_idx] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
                self.optimized_texturecodes[data_idx][obj_idx] = torch.zeros((n_code, latent_dim), dtype=torch.float32)

        # initialize evaluation scores
        self.psnr_eval = {}
        self.psnr_opt = {}
        self.ssim_eval = {}
        self.R_eval = {}
        self.T_eval = {}
        self.depth_err_mean = {}
        self.lidar_pts_cnt = {}

    def run(self):
        if self.opt_pose:
            if self.hpams_pose_refiner is not None or self.hpams['arch'] == 'supnerf':
                self.optimize_objs_w_pose_unified()
            else:
                self.optimize_objs_w_pose()
        else:
            self.optimize_objs()

    def optimize_objs(self):
        """
            Optimize on each annotation frame independently
        """
    
        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')
    
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            obj_poses = batch_data['obj_poses']
            wlh_batch = batch_data['wlh']
    
            # convert obj pose from waymo to nusc then all the process afterward is same
            obj_poses = obj_pose_kitti2nusc(obj_poses, wlh_batch[:, 2])
            cam_R = obj_poses[0, :, :3]
            cam_T = obj_poses[0, :, 3:]
            tgt_cam = torch.cat([cam_R.transpose(-2, -1), -cam_R.transpose(-2, -1) @ cam_T], dim=-1)
    
            tgt_img, mask_occ, roi, K = \
                imgs[0], masks_occ[0], rois[0], cam_intrinsics[0]
    
    
            data_idx, obj_idx = batch_data['data_idx'][0], batch_data['obj_idx'][0]
            log_idx = f'{data_idx}_{obj_idx}'
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
    
            H, W = tgt_img.shape[0:2]
            roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
    
            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)
    
            if 'autorf' in self.hpams['arch']:
                # preprocess image and predict shapecode and texturecode
                if self.use_bn:
                    img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                else:
                    img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
    
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')
    
            # First Optimize
            self.nopts = 0
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[data_idx][obj_idx][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[data_idx][obj_idx][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                loss_per_img = []
    
                # render ray values and prepare target rays
                rgb_rays, depth_pred_vec, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                        tgt_img, mask_occ, tgt_cam,
                                                                                        obj_diag, K, roi,
                                                                                        self.hpams['n_rays'],
                                                                                        self.hpams['n_samples'],
                                                                                        shapecode, texturecode,
                                                                                        self.hpams['shapenet_obj_cood'],
                                                                                        self.hpams['sym_aug'])
    
                # Compute losses
                loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                            torch.sum(torch.abs(occ_pixels)) + 1e-9)
                loss_occ = torch.sum(
                    torch.exp(-occ_pixels * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels)) / (
                                       torch.sum(torch.abs(occ_pixels)) + 1e-9)
                # loss_reg = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                loss = loss_rgb + self.hpams['loss_occ_coef'] * loss_occ
                loss.backward()
    
                # save the rgb loss for psnr computation (only include object mask)
                mask_rgb = occ_pixels.clone()
                mask_rgb[occ_pixels < 0] = 0
                loss_rgb2 = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb) + 1e-9)
                loss_per_img.append(loss_rgb2.detach().item())
                # if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                self.log_eval_psnr(loss_per_img, [log_idx])
                # log depth error every iter only rendering the lidar pixels
                gt_depth_map = batch_data['depth_maps'][0, roi[1]:roi[3], roi[0]:roi[2]].numpy()
                y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                gt_depth_vec = gt_depth_map[y_vec, x_vec]
                with torch.no_grad():
                    _, depth_pred_vec, _, _, _ = render_rays_specified(self.model, self.device,
                                                                       tgt_img, mask_occ, tgt_cam,
                                                                       obj_diag, K, roi, x_vec, y_vec,
                                                                       self.hpams['n_samples'],
                                                                       shapecode, texturecode,
                                                                       self.hpams['shapenet_obj_cood'],
                                                                       self.hpams['sym_aug'])
                self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, log_idx)
                self.opts.step()
    
                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    gt_imgs = []
                    gt_masks_occ = []
                    gt_depth_maps = []
                    gt_imgs.append(imgs[0, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[0, roi[1]:roi[3], roi[0]:roi[2]])
                    gt_depth_maps.append(batch_data['depth_maps'][0, roi[1]:roi[3], roi[0]:roi[2]].numpy())
    
                    # generate the full images
                    generated_imgs = []
                    generated_depth_maps = []
                    with torch.no_grad():
                        # render full image
                        generated_img, generated_depth = render_full_img(self.model, self.device, tgt_cam, obj_sz, K, roi,
                                                        self.hpams['n_samples'], shapecode, texturecode,
                                                        self.hpams['shapenet_obj_cood'], out_depth=True)
                        generated_imgs.append(generated_img)
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, data_idx, self.nopts)
                        generated_depth_maps.append(generated_depth.cpu().numpy())
                        # # save virtual views at the beginning and the end
                        # if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                        #     virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                        #                                        self.hpams['n_samples'], shapecode, texturecode,
                        #                                        self.hpams['shapenet_obj_cood'])
                        #     self.save_virtual_img(virtual_imgs, data_idx, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers(shapecode, texturecode)
    
            # Save the optimized codes
            self.optimized_shapecodes[data_idx][obj_idx][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[data_idx][obj_idx][-1] = texturecode.detach().cpu()
            if batch_idx % self.save_freq == 0 or batch_idx == (len(self.dataloader) - 1):
                print(f'save result at batch {batch_idx}')
                self.save_opts(batch_idx)

    def optimize_objs_w_pose(self):
        """
            Optimize on each annotation frame independently
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            obj_poses = batch_data['obj_poses']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            wlh_batch = batch_data['wlh']

            # convert obj pose from waymo to nusc then all the process afterward is same
            obj_poses = obj_pose_kitti2nusc(obj_poses, wlh_batch[:, 2])
            obj_poses_w_err = obj_pose_kitti2nusc(obj_poses_w_err, wlh_batch[:, 2])

            tgt_img, pred_pose, mask_occ, roi, K = \
                imgs[0], obj_poses_w_err[0], masks_occ[0], rois[0], cam_intrinsics[0]

            data_idx, obj_idx = batch_data['data_idx'][0], batch_data['obj_idx'][0]
            log_idx = f'{data_idx}_{obj_idx}'
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_img.shape[0:2]
            roi_new2 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)

            # crop tgt img to roi
            tgt_img = tgt_img[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]]
            mask_occ = mask_occ[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)

            if 'autorf' in self.hpams['arch']:
                # preprocess image and predict shapecode and texturecode
                if self.use_bn:
                    img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                else:
                    img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

            # If to use external pose regressor, apply here (assume nerf arch is autorf or supnerf)
            if self.hpams_pose_regressor is not None and self.opt_pose == 4:
                _, pred_pose_vec = self.model_pose_regressor.im_encode(img_in.to(self.device))
                pred_rot_mat = rot_trans.axis_angle_to_matrix(pred_pose_vec[:, :3])
                pred_pose = torch.cat([pred_rot_mat, pred_pose_vec[:, 3:].unsqueeze(-1)], dim=-1)[0]

            # set pose parameters
            rot_mat_vec = pred_pose[:3, :3].unsqueeze(0)
            Z_init = pred_pose[2, 3].clone().to(self.device)
            trans_vec = pred_pose[:3, 3].unsqueeze(0).to(self.device).detach().requires_grad_()
            if self.hpams['euler_rot']:
                rot_vec = rot_trans.matrix_to_euler_angles(rot_mat_vec, 'XYZ').to(self.device).detach().requires_grad_()
            else:
                rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimization
            self.nopts = 0
            # self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            if self.opt_pose == 0:
                self.set_optimizers(shapecode, texturecode)
            else:
                self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[data_idx][obj_idx][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[data_idx][obj_idx][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                loss_per_img = []

                t2opt = trans_vec[0].unsqueeze(-1)
                if self.hpams['euler_rot']:
                    rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[0], 'XYZ')
                else:
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])

                if not self.hpams['optimize']['opt_cam_pose']:
                    rot_mat2opt = rot_mat2opt.transpose(-2, -1)
                    t2opt = -rot_mat2opt @ t2opt

                cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                           tgt_img, mask_occ, cam2opt,
                                                                                           obj_diag, cam_intrinsics[0],
                                                                                           roi_new2,
                                                                                           self.hpams['n_samples'],
                                                                                           shapecode, texturecode,
                                                                                           self.hpams[
                                                                                               'shapenet_obj_cood'],
                                                                                           self.hpams['sym_aug'],
                                                                                           im_sz=self.hpams[
                                                                                               'render_im_sz'],
                                                                                           n_rays=None)
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
                psnr = self.log_eval_psnr(loss_per_img, [log_idx])

                # save pose loss
                pred_obj_R = cam2opt[:, :3].detach().cpu().transpose(-2, -1)
                pred_obj_T = -pred_obj_R @ (cam2opt[:, 3:].detach().cpu())
                pred_obj_poses = torch.cat([pred_obj_R, pred_obj_T], dim=-1).unsqueeze(0)
                errs_R, errs_T = self.log_eval_pose(pred_obj_poses, obj_poses, [log_idx])

                # save depth error every iter only rendering the lidar pixels
                gt_depth_map = batch_data['depth_maps'][0, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2]].numpy()
                y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                gt_depth_vec = gt_depth_map[y_vec, x_vec]
                with torch.no_grad():
                    _, depth_pred_vec, _, _, _ = render_rays_specified(self.model, self.device,
                                                                       tgt_img, mask_occ, cam2opt,
                                                                       obj_diag, K, roi_new2, x_vec, y_vec,
                                                                       self.hpams['n_samples'],
                                                                       shapecode, texturecode,
                                                                       self.hpams['shapenet_obj_cood'],
                                                                       self.hpams['sym_aug'])
                depth_err = self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, log_idx)
                # depth_err = 0
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    gt_imgs = []
                    gt_masks_occ = []
                    gt_depth_maps = []
                    gt_imgs.append(imgs[0, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[0, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2]])
                    gt_depth_maps.append(batch_data['depth_maps'][0, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2]].numpy())

                    # generate the full images
                    cam2opt = cam2opt.cpu().detach()
                    with torch.no_grad():
                        rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                tgt_img, mask_occ, cam2opt,
                                                                                obj_diag, cam_intrinsics[0],
                                                                                roi_new2, self.hpams['n_samples'],
                                                                                shapecode, texturecode,
                                                                                self.hpams['shapenet_obj_cood'],
                                                                                self.hpams['sym_aug'],
                                                                                im_sz=vis_im_sz, n_rays=None)
                        generated_img = rgb_rays.view((vis_im_sz, vis_im_sz, 3))
                        generated_depth = depth_rays.view((vis_im_sz, vis_im_sz))
                        gt_img = imgs[:, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2], :].cpu()
                        gt_img = gt_img.permute((0, 3, 1, 2))
                        gt_img = Resize((vis_im_sz, vis_im_sz))(gt_img).permute((0, 2, 3, 1))[0]
                        gt_occ = occ_pixels.view((vis_im_sz, vis_im_sz)).cpu()

                        # draw 3D box given predicted pose
                        est_R = cam2opt[:, :3].numpy().T
                        est_T = -est_R @ cam2opt[:, 3:].numpy()
                        pred_uv = view_points(
                            corners_of_box(np.concatenate([est_R, est_T], axis=1), wlh_batch[0].numpy()),
                            K.numpy(), normalize=True)
                        pred_uv[0, :] -= roi_new2[0].item()
                        pred_uv[1, :] -= roi_new2[1].item()

                        u_vec_new, v_vec_new = roi_coord_trans(pred_uv[0, :], pred_uv[1, :],
                                                               roi_new2.numpy(), im_sz_tgt=vis_im_sz)
                        pred_uv[0, :] = u_vec_new
                        pred_uv[1, :] = v_vec_new

                        # generated_img = render_box(generated_img.cpu().numpy(), pred_uv, colors=(g_c, g_c, g_c))
                        gt_img = render_box(gt_img.cpu().numpy(), pred_uv, colors=(box_c, box_c, box_c))

                        self.save_img3([generated_img],
                                       [generated_depth],
                                       [torch.from_numpy(gt_img)], log_idx, self.nopts,
                                       psnr, [depth_err], errs_R, errs_T)

                        # # save virtual views at the beginning and the end
                        # if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                        #     virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                        #                                        self.hpams['n_samples'], shapecode, texturecode,
                        #                                        self.hpams['shapenet_obj_cood'])
                        #     self.save_virtual_img(virtual_imgs, data_idx, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    if self.opt_pose == 0:
                        self.set_optimizers(shapecode, texturecode)
                    else:
                        self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[data_idx][obj_idx][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[data_idx][obj_idx][-1] = texturecode.detach().cpu()
            if batch_idx % self.save_freq == 0 or batch_idx == (len(self.dataloader) - 1):
                print(f'save result at batch {batch_idx}')
                self.save_opts_w_pose(batch_idx)

    def dl_pose_update(self,
                       im_feat_batch,
                       src_pose_batch,
                       wlh_batch,
                       roi_batch,
                       K_batch,
                       K_inv_batch,
                       iters=3,
                       start_wt_est_pose=False,
                       pred_uv_batch_direct=None,
                       ):
        """
            The input pose need to align with the training model.
            The pose is converted to trained coordinate system before fed in
        """
        pose_per_iter = []

        if start_wt_est_pose:
            pred_uv_batch_direct = pred_uv_batch_direct.view((-1, 2, 8))
            # convert to original image frame
            dim_batch = torch.maximum(roi_batch[:, 2] - roi_batch[:, 0], roi_batch[:, 3] - roi_batch[:, 1])
            pred_uv_batch_direct *= (dim_batch.view((-1, 1, 1)) / 2)
            pred_uv_batch_direct[:, 0, :] += ((roi_batch[:, 0:1] + roi_batch[:, 2:3]) / 2)
            pred_uv_batch_direct[:, 1, :] += ((roi_batch[:, 1:2] + roi_batch[:, 3:4]) / 2)

            # run pnp to estimate pose
            p2d_np = pred_uv_batch_direct[0].cpu().numpy().T
            w, l, h = wlh_batch[0].cpu().numpy()
            # 3D bounding box corners. (Convention: x forward, y left, z up.)
            x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
            z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
            p3d_np = np.vstack((x_corners, y_corners, z_corners)).T
            dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                p3d_np, p2d_np, K_batch[0].cpu().numpy(), dist_coeff,
                iterationsCount=5000,
                reprojectionError=1,  # this depends on the coordinate frame of 2D pts
                flags=cv2.SOLVEPNP_P3P)
            if rvec is not None and tvec is not None:
                R_est, _ = cv2.Rodrigues(rvec)
                t_est = tvec
                src_pose_batch[0, :3, :3] = torch.from_numpy(R_est).to(self.device)
                # Ignore out-of-distribution t_est?
                if tvec[2] > 0 and tvec[2] < 60:
                    src_pose_batch[0, :3, 3:4] = torch.from_numpy(t_est).to(self.device)
            else:
                print(f'matches lead to no transformation')

        pose_per_iter.append(src_pose_batch.squeeze().cpu().numpy())
        for iter in range(0, iters):
            src_pose_batch = self.fw_pose_one_step(
                im_feat_batch,
                src_pose_batch,
                wlh_batch,
                roi_batch,
                K_batch,
                K_inv_batch,
                nusc2waymo_boxfac=self.nusc2waymo_boxfac
            )
            pose_per_iter.append(src_pose_batch.squeeze().cpu().numpy())
        return pose_per_iter

    def fw_pose_one_step(self,
                         im_feat_batch,
                         src_pose_batch,
                         wlh_batch,
                         roi_batch,
                         K_batch,
                         K_inv_batch,
                         nusc2waymo_boxfac=1.1
                         ):

        # TODO: change the hand-coded scale for auto fit to nuscanes
        src_uv_batch = view_points_batch(corners_of_box_batch(src_pose_batch, wlh_batch, scale=nusc2waymo_boxfac),
                                         K_batch, normalize=True)

        # normalize src_uv_batch to align with img_in_batch frame, now normalized to (-1, 1) x (-1, 1)
        src_uv_norm, dim_batch = normalize_by_roi(src_uv_batch[:, :2, :], roi_batch, need_square=True)

        # regress delta_pose may be better than delta uv which may not consistent which is more dependent on ransac
        bsize = im_feat_batch.shape[0]
        if self.hpams['arch'] == 'supnerf':
            delta_pose_batch = self.model.pose_update(im_feat_batch, src_uv_norm.view((bsize, -1)))
        else:
            delta_pose_batch = self.model_pose_refiner.pose_update(im_feat_batch, src_uv_norm.view((bsize, -1)))

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

    def optimize_objs_w_pose_unified(self):
        """
            Optimize on each annotation frame independently
            Use additional pose refiner network to approach true pose iteratively
            If only to regress pose, the nerf model is only for the rendering and visualization purpose

            Version 2: use fixed sized ROI for rendering
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'num obj: {batch_idx}/{len(self.dataloader)}')
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            # cam_poses = batch_data['cam_poses']
            obj_poses = batch_data['obj_poses']
            # cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            wlh_batch = batch_data['wlh']
            K_inv_batch = torch.linalg.inv(cam_intrinsics)

            # convert obj pose from waymo to nusc then all the process afterward is same
            obj_poses = obj_pose_kitti2nusc(obj_poses, wlh_batch[:, 2])
            obj_poses_w_err = obj_pose_kitti2nusc(obj_poses_w_err, wlh_batch[:, 2])

            tgt_img, pred_pose, mask_occ, roi = \
                imgs[0], obj_poses_w_err[0], masks_occ[0], rois[0]

            data_idx, obj_idx = batch_data['data_idx'][0], batch_data['obj_idx'][0]
            log_idx = f'{data_idx}_{obj_idx}'
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_img.shape[0:2]
            # pad roi to make square
            roi_new1 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)

            # crop tgt img to roi
            tgt_img = tgt_img[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]]
            mask_occ = mask_occ[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)

            # preprocess image and
            img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
            if self.hpams['arch'] == 'supnerf':
                if self.hpams['net_hyperparams']['pred_wlh'] > 0 and self.pred_wlh:
                    shapecode, texturecode, posecode, pred_uv_batch_direct, wlh_batch = self.model.encode_img(
                        img_in.to(self.device))
                    wlh_batch = wlh_batch.detach().cpu()
                    if self.pred_wlh == 2:  # only use part of estimation, the rest from mean over training
                        wlh_batch_new = wlh_batch.clone()
                        wlh_batch_new[:, 0] = WLH_MEAN[0]
                        # wlh_batch_new[:, 1] = WLH_MEAN[1]
                        wlh_batch_new[:, 2] = WLH_MEAN[2]
                        wlh_batch_new[:, 1] = wlh_batch[:, 0] * wlh_batch[:, 1] * wlh_batch[:, 2] / wlh_batch_new[:, 0] / wlh_batch_new[:, 2]
                        wlh_batch = wlh_batch_new
                    obj_sz = wlh_batch[0].numpy()
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
                else:
                    shapecode, texturecode, posecode, pred_uv_batch_direct, _ = self.model.encode_img(img_in.to(self.device))
                # Use the mean to improve completeness
                shapecode = (shapecode + self.mean_shape.clone().to(self.device)) / 2
                texturecode = (texturecode + self.mean_texture.clone().to(self.device)) / 2
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            elif 'autorf' in self.hpams['arch']:
                # predict shapecode and texturecode
                shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                posecode, pred_uv_batch_direct = self.model_pose_refiner.im_encode(img_in.to(self.device))
                # Use the mean to improve completeness
                shapecode = (shapecode + self.mean_shape.clone().to(self.device))/2
                texturecode = (texturecode + self.mean_texture.clone().to(self.device))/2
                shapecode = shapecode.detach().requires_grad_()
                texturecode = texturecode.detach().requires_grad_()
            else:  # 'codenerf case'
                posecode, pred_uv_batch_direct = self.model_pose_refiner.im_encode(img_in.to(self.device))
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()

            """
                Pose Refine before NeRF
            """
            with torch.no_grad():
                pose_per_iter = self.dl_pose_update(
                    posecode,
                    obj_poses_w_err.to(self.device),
                    wlh_batch.to(self.device),
                    roi_new1.unsqueeze(0).to(self.device),
                    cam_intrinsics.to(self.device),
                    K_inv_batch.to(self.device),
                    iters=self.reg_iters,
                    start_wt_est_pose=self.opt_pose == 2,
                    pred_uv_batch_direct=pred_uv_batch_direct)
            pose_per_iter = torch.from_numpy(np.array(pose_per_iter))
            pose_per_iter = pose_per_iter.to(self.device)
            pred_pose = pose_per_iter[-1]

            """
                Optimization NeRF
            """

            # set pose parameters
            rot_mat_vec = pred_pose[:3, :3].unsqueeze(0)
            Z_init = pred_pose[2, 3].clone().to(self.device)
            trans_vec = pred_pose[:3, 3].unsqueeze(0).to(self.device).detach().requires_grad_()
            rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            self.nopts = 0
            if self.opt_pose == 0:
                self.set_optimizers(shapecode, texturecode)
            else:
                self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[data_idx][obj_idx][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[data_idx][obj_idx][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                loss_per_img = []

                # the first a few to load pre-computed poses for evaluation
                if self.nopts > self.reg_iters:
                    t2opt = trans_vec[0].unsqueeze(-1)
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])
                else:
                    t2opt = pose_per_iter[self.nopts, :3, 3:]
                    rot_mat2opt = pose_per_iter[self.nopts, :3, :3]

                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_poses[data_idx][obj_idx][code_i] = torch.cat((rot_mat2opt, t2opt), dim=-1).detach().cpu()

                # convert from o2c to c2o for nerf use
                rot_mat2opt = rot_mat2opt.transpose(-2, -1)
                t2opt = -rot_mat2opt @ t2opt
                cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # TODO: A different roi and crop for nerf (Can be removed when new model is trained for new infer)
                tgt_img = imgs[0]
                mask_occ = masks_occ[0]
                H, W = tgt_img.shape[0:2]
                roi_new2 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)

                # crop tgt img to roi
                tgt_img = tgt_img[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]]
                mask_occ = mask_occ[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]].unsqueeze(-1)
                # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                tgt_img = tgt_img * (mask_occ > 0)
                tgt_img = tgt_img + (mask_occ <= 0)

                # render ray values and prepare target rays
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                           tgt_img, mask_occ, cam2opt,
                                                                                           obj_diag, cam_intrinsics[0],
                                                                                           roi_new2, self.hpams['n_samples'],
                                                                                           shapecode, texturecode,
                                                                                           self.hpams[
                                                                                               'shapenet_obj_cood'],
                                                                                           self.hpams['sym_aug'],
                                                                                           im_sz=self.hpams['render_im_sz'],
                                                                                           n_rays=None)
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
                psnr = self.log_eval_psnr(loss_per_img, [log_idx])

                # save pose loss
                pred_obj_R = cam2opt[:, :3].detach().cpu().transpose(-2, -1)
                pred_obj_T = -pred_obj_R @ (cam2opt[:, 3:].detach().cpu())
                pred_obj_poses = torch.cat([pred_obj_R, pred_obj_T], dim=-1).unsqueeze(0)
                errs_R, errs_T = self.log_eval_pose(pred_obj_poses, obj_poses, [log_idx])

                # log depth error every iter only rendering the lidar pixels
                gt_depth_map = batch_data['depth_maps'][0, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2]].numpy()
                y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                gt_depth_vec = gt_depth_map[y_vec, x_vec]

                with torch.no_grad():
                    _, depth_pred_vec, _, _, _ = render_rays_specified(self.model, self.device,
                                                                       tgt_img, mask_occ, cam2opt,
                                                                       obj_diag, cam_intrinsics[0], roi_new2, x_vec, y_vec,
                                                                       self.hpams['n_samples'],
                                                                       shapecode, texturecode,
                                                                       self.hpams['shapenet_obj_cood'],
                                                                       self.hpams['sym_aug'])
                depth_err = self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, log_idx)
                # depth_err = 0

                # The first a few is just for evaluation purpose to render rays, do not update codes or pose
                if self.nopts > self.reg_iters:
                    self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts <= self.reg_iters or
                                                        self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    cam2opt = cam2opt.cpu().detach()
                    with torch.no_grad():
                        rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                tgt_img, mask_occ, cam2opt,
                                                                                obj_diag, cam_intrinsics[0],
                                                                                roi_new2, self.hpams['n_samples'],
                                                                                shapecode, texturecode,
                                                                                self.hpams['shapenet_obj_cood'],
                                                                                self.hpams['sym_aug'],
                                                                                im_sz=vis_im_sz, n_rays=None)
                        generated_img = rgb_rays.view((vis_im_sz, vis_im_sz, 3))
                        generated_depth = depth_rays.view((vis_im_sz, vis_im_sz))
                        gt_img = imgs[:, roi_new2[1]:roi_new2[3], roi_new2[0]:roi_new2[2], :].cpu()
                        gt_img = gt_img.permute((0, 3, 1, 2))
                        gt_img = Resize((vis_im_sz, vis_im_sz))(gt_img).permute((0, 2, 3, 1))[0]
                        gt_occ = occ_pixels.view((vis_im_sz, vis_im_sz)).cpu()

                        # draw 3D box given predicted pose
                        est_R = cam2opt[:, :3].numpy().T
                        est_T = -est_R @ cam2opt[:, 3:].numpy()
                        pred_uv = view_points(
                            corners_of_box(np.concatenate([est_R, est_T], axis=1), wlh_batch[0].numpy()),
                            cam_intrinsics[0].numpy(), normalize=True)
                        pred_uv[0, :] -= roi_new2[0].item()
                        pred_uv[1, :] -= roi_new2[1].item()

                        u_vec_new, v_vec_new = roi_coord_trans(pred_uv[0, :], pred_uv[1, :],
                                                                    roi_new2.numpy(), im_sz_tgt=vis_im_sz)
                        pred_uv[0, :] = u_vec_new
                        pred_uv[1, :] = v_vec_new
                        # generated_img = render_box(generated_img.cpu().numpy(), pred_uv, colors=(g_c, g_c, g_c))
                        gt_img = render_box(gt_img.cpu().numpy(), pred_uv, colors=(box_c, box_c, box_c))

                        self.save_img3([generated_img],
                                       [generated_depth],
                                       [torch.from_numpy(gt_img)], log_idx, self.nopts,
                                       psnr, [depth_err], errs_R, errs_T)

                        # # save virtual views at the beginning and the end
                        # if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                        #     virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                        #                                        self.hpams['n_samples'], shapecode, texturecode,
                        #                                        self.hpams['shapenet_obj_cood'], img_sz=64)
                        #     self.save_virtual_img(virtual_imgs, log_idx, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    if self.opt_pose == 0:
                        self.set_optimizers(shapecode, texturecode)
                    else:
                        self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            # Save the optimized codes
            self.optimized_shapecodes[data_idx][obj_idx][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[data_idx][obj_idx][-1] = texturecode.detach().cpu()
            self.optimized_poses[data_idx][obj_idx][-1] = pred_obj_poses[0].detach().cpu()
            if batch_idx % self.save_freq == 0 or batch_idx == (len(self.dataloader) - 1):
                print(f'save result at batch {batch_idx}')
                self.save_opts_w_pose(batch_idx)

    def save_opts(self, num_obj):
        saved_dict = {
            'num_obj': num_obj,
            'optimized_shapecodes': self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval,
            'depth_err_mean': self.depth_err_mean,
            'lidar_pts_cnt': self.lidar_pts_cnt,
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        # print('We finished the optimization of object' + str(num_obj))

    def save_opts_w_pose(self, num_obj):
        saved_dict = {
            'num_obj': num_obj,
            'optimized_shapecodes': self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'optimized_poses': self.optimized_poses,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval,
            'depth_err_mean': self.depth_err_mean,
            'lidar_pts_cnt': self.lidar_pts_cnt,
            'R_eval': self.R_eval,
            'T_eval': self.T_eval,
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes+poses.pth'))
        # print('We finished the optimization of ' + str(num_obj))

    # TODO: add lidar visual
    def save_img(self, generated_imgs, gt_imgs, masks_occ, obj_id, instance_num):
        # H, W = gt_imgs[0].shape[:2]
        W_tgt = np.min([gt_img.shape[1] for gt_img in gt_imgs])

        if len(gt_imgs) > 1:
            # Align the width of different-sized images
            generated_imgs = align_imgs_width(generated_imgs, W_tgt)
            # masks_occ = align_imgs_width(masks_occ, W_tgt)
            gt_imgs = align_imgs_width(gt_imgs, W_tgt)

        generated_imgs = torch.cat(generated_imgs).reshape(-1, W_tgt, 3)
        # masks_occ = torch.cat(masks_occ).reshape(-1, W_tgt, 1)
        gt_imgs = torch.cat(gt_imgs).reshape(-1, W_tgt, 3)
        H_cat = generated_imgs.shape[0]

        ret = torch.zeros(H_cat, 2 * W_tgt, 3)
        ret[:, :W_tgt, :] = generated_imgs.reshape(-1, W_tgt, 3)
        # ret[:, W_tgt:, :] = gt_imgs.reshape(-1, W_tgt, 3) * 0.75 + masks_occ.reshape(-1, W_tgt, 1) * 0.25
        ret[:, W_tgt:, :] = gt_imgs.reshape(-1, W_tgt, 3)
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(instance_num) + '.png'), ret)

    def save_img3(self, gen_rgb_imgs, gen_depth_imgs, gt_imgs, obj_id, instance_num, psnr, depth_err, rot_err, trans_err):
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
        err_str1 = 'PSNR: {:.3f},  DE: {:.3f}'.format(
            psnr[0], depth_err[0])
        err_str2 = 'RE: {:.3f},  TE: {:.3f}'.format(
            rot_err[0], trans_err[0])

        ret = cv2.putText(ret, err_str1, (int(5 * puttext_ratio), int(10 * puttext_ratio)), cv2.FONT_HERSHEY_SIMPLEX,
                          .35 * puttext_ratio, (0, 0, 0), thickness=int(puttext_ratio))
        ret = cv2.putText(ret, err_str2, (int(5 * puttext_ratio), int(21 * puttext_ratio)), cv2.FONT_HERSHEY_SIMPLEX,
                          .35 * puttext_ratio, (0, 0, 0), thickness=int(puttext_ratio))

        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(instance_num) + '.png'), ret)

    def save_virtual_img(self, imgs, obj_id, instance_num=None):
        H, W = imgs[0].shape[:2]

        img_out = torch.cat(imgs).reshape(-1, W, 3)
        img_out = image_float_to_uint8(img_out.detach().cpu().numpy())
        img_out = np.concatenate([img_out[:4*H, ...], img_out[4*H:, ...]], axis=1)
        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if instance_num is None:
            imageio.imwrite(os.path.join(save_img_dir, 'virt_final.png'), img_out)
        else:
            imageio.imwrite(os.path.join(save_img_dir, 'virt_opt' + '{:03d}'.format(instance_num) + '.png'), img_out)

    def log_compute_ssim(self, generated_imgs, gt_imgs, ann_tokens):
        # ATTENTION: preparing whole images is time-consuming for evaluation purpose
        for i, ann_token in enumerate(ann_tokens):
            generated_img_np = generated_imgs[i].detach().cpu().numpy()
            gt_img_np = gt_imgs[i].detach().cpu().numpy()
            ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
            if self.ssim_eval.get(ann_token) is None:
                self.ssim_eval[ann_token] = [ssim]
            else:
                self.ssim_eval[ann_token].append(ssim)

    def log_eval_psnr(self, loss_per_img, ann_tokens):
        # ATTENTION: the loss_per_img should only include foreground object mask
        psnr_list = []
        for i, ann_token in enumerate(ann_tokens):
            psnr = -10 * np.log(loss_per_img[i]) / np.log(10)
            psnr_list.append(psnr)
            if self.psnr_eval.get(ann_token) is None:
                self.psnr_eval[ann_token] = [psnr]
            else:
                self.psnr_eval[ann_token].append(psnr)

            if self.nopts == 0:
                print('   Initial psnr: {:.3f}'.format(psnr))
            if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                print('   Final psnr: {:.3f}'.format(psnr))
        return np.array(psnr_list)

    def log_eval_pose(self, est_poses, tgt_poses, ann_tokens):
        """
            input need to be object pose
            TODO: should record actual R T for later use
        """
        err_R, err_T = calc_pose_err(est_poses, tgt_poses)

        for i, ann_token in enumerate(ann_tokens):
            if self.R_eval.get(ann_token) is None:
                self.R_eval[ann_token] = [err_R[i]]
                self.T_eval[ann_token] = [err_T[i]]
            else:
                self.R_eval[ann_token].append(err_R[i])
                self.T_eval[ann_token].append(err_T[i])
            if math.isnan(err_T[i]) or math.isnan(err_R[i]):
                print('FOUND NaN')
            if self.nopts == 0:
                print('   Initial RE: {:.3f}, TE: {:.3f}'.format(err_R[i], err_T[i]))
            if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                print('   Final RE: {:.3f}, TE: {:.3f}'.format(err_R[i], err_T[i]))
        return err_R, err_T

    def log_eval_depth(self, pred_depth_maps, gt_depth_maps, gt_occ_maps, ann_tokens, debug=False):
        depth_err_list = []
        for i, ann_token in enumerate(ann_tokens):
            lidar_pts_mask = np.logical_and(gt_depth_maps[i] > 0, gt_occ_maps[i].numpy() > 0)
            pred_depth_map = pred_depth_maps[i][lidar_pts_mask]
            gt_depth_map = gt_depth_maps[i][lidar_pts_mask]
            depth_errs = abs(pred_depth_map - gt_depth_map)

            if debug:
                fig, axes = plt.subplots(3, 1, figsize=(9, 27))
                axes[0].imshow(pred_depth_maps[i])
                axes[0].set_title('pred depth')
                axes[1].imshow(gt_depth_maps[i])
                axes[1].set_title('gt depth')
                axes[2].imshow(gt_occ_maps[i] > 0)
                axes[2].set_title('fg mask')
                plt.show()

            self.depth_err_mean[ann_token] = np.sum(depth_errs) / (np.sum(lidar_pts_mask) + 1e-8)
            self.lidar_pts_cnt[ann_token] = np.sum(lidar_pts_mask)
            print(f'    View: {i+1}, depth error mean: {self.depth_err_mean[ann_token]:.3f}, total lidar pts: {self.lidar_pts_cnt[ann_token]}')
            depth_err_list.append(self.depth_err_mean[ann_token])
        return np.array(depth_err_list)

    def log_eval_depth_v2(self, depth_rays, gt_depth_vec, ann_token):
        """
            More efficient evaluation at sparse pixels associated with lidar measurements.
        """
        depth_err_list = []
        depth_errs = abs(depth_rays - gt_depth_vec)
        self.lidar_pts_cnt[ann_token] = len(gt_depth_vec)
        depth_err_mean = np.sum(depth_errs) / (len(gt_depth_vec) + 1e-8)

        if self.depth_err_mean.get(ann_token) is None:
            self.depth_err_mean[ann_token] = [depth_err_mean]
        else:
            self.depth_err_mean[ann_token].append(depth_err_mean)

        if self.nopts == 0:
            print(f'   Initial depth err: {depth_err_mean:.3f}, total lidar pts: {self.lidar_pts_cnt[ann_token]}')
        if self.nopts == self.hpams['optimize']['num_opts'] - 1:
            print(f'   Final depth err: {depth_err_mean:.3f}, total lidar pts: {self.lidar_pts_cnt[ann_token]}')
        return depth_err_mean

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
            {'params': trans, 'lr':  self.hpams['optimize']['lr_pose']}
        ])

    def update_learning_rate(self):
        opt_values = self.nopts // self.hpams['optimize']['lr_half_interval']
        self.hpams['optimize']['lr_shape'] = self.hpams['optimize']['lr_shape'] * 2**(-opt_values)
        self.hpams['optimize']['lr_texture'] = self.hpams['optimize']['lr_texture'] * 2**(-opt_values)
        self.hpams['optimize']['lr_pose'] = self.hpams['optimize']['lr_pose'] * 2**(-opt_values)

    def make_model(self):
        if self.hpams['arch'] == 'autorf':
            self.model = AutoRF(**self.hpams['net_hyperparams']).to(self.device)
        if self.hpams['arch'] == 'autorfmix':
            self.model = AutoRFMix(**self.hpams['net_hyperparams']).to(self.device)
        elif self.hpams['arch'] == 'codenerf':
            self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)
        elif self.hpams['arch'] == 'supnerf':
            self.model = SUPNeRF(**self.hpams['net_hyperparams']).to(self.device)
        else:
            print('ERROR: No valid network architecture is declared in config file!')

    def load_model(self, epoch_load=None):
        saved_dir = self.hpams['model_dir']
        if epoch_load is None:
            saved_path = os.path.join(saved_dir, 'models.pth')
        else:
            saved_path = os.path.join(saved_dir, f'epoch_{epoch_load}.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))
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

        if self.hpams_pose_refiner is not None:
            saved_dir2 = self.hpams_pose_refiner['model_dir']
            if epoch_load is None:
                saved_path2 = os.path.join(saved_dir2, 'models.pth')
            else:
                saved_path2 = os.path.join(saved_dir2, f'epoch_{epoch_load}.pth')
            saved_data2 = torch.load(saved_path2, map_location=torch.device('cpu'))
            self.model_pose_refiner.load_state_dict(saved_data2['model_params'])
            self.model_pose_refiner = self.model_pose_refiner.to(self.device)

        if self.hpams_pose_regressor is not None:
            saved_dir2 = self.hpams_pose_regressor['model_dir']
            # if epoch_load is None:
            saved_path2 = os.path.join(saved_dir2, 'models.pth')
            # else:
            #     saved_path2 = os.path.join(saved_dir2, f'epoch_{epoch_load}.pth')
            saved_data2 = torch.load(saved_path2, map_location=torch.device('cpu'))
            self.model_pose_regressor.load_state_dict(saved_data2['model_params'])
            self.model_pose_regressor = self.model_pose_regressor.to(self.device)

    def make_save_img_dir(self):
        if self.hpams_pose_refiner is not None:
            save_dir_tmp = self.hpams_pose_refiner['model_dir'] + '/test' + self.save_postfix
        elif self.hpams_pose_regressor is not None:
            save_dir_tmp = self.hpams_pose_regressor['model_dir'] + '/test' + self.save_postfix
        else:
            save_dir_tmp = self.hpams['model_dir'] + '/test' + self.save_postfix
        if self.model_epoch is not None:
            save_dir_tmp += f'_epoch_{self.model_epoch}'
        if self.vis > 0:
            save_dir_tmp += f'_vis{self.vis}'

        num = 2
        while os.path.isdir(save_dir_tmp):
            if self.hpams_pose_refiner is not None:
                save_dir_tmp = self.hpams_pose_refiner['model_dir'] + '/test' + self.save_postfix
            elif self.hpams_pose_regressor is not None:
                save_dir_tmp = self.hpams_pose_regressor['model_dir'] + '/test' + self.save_postfix
            else:
                save_dir_tmp = self.hpams['model_dir'] + '/test' + self.save_postfix
            if self.model_epoch is not None:
                save_dir_tmp += f'_epoch_{self.model_epoch}'
            if self.vis > 0:
                save_dir_tmp += f'_vis{self.vis}'
            save_dir_tmp += '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
