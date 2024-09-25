import copy
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

from utils import image_float_to_uint8, generate_obj_sz_reg_samples, align_imgs_width, render_rays, render_rays_v2, \
    render_rays_specified, render_full_img, render_virtual_imgs, view_points_batch, corners_of_box_batch, \
    calc_pose_err,  preprocess_img_square, preprocess_img_keepratio, normalize_by_roi, render_box, \
    view_points, corners_of_box, roi_coord_trans, roi_process, colorize
from skimage.metrics import structural_similarity as compute_ssim
from model_autorf import AutoRF, AutoRFMix
from model_codenerf import CodeNeRF
from model_supnerf import SUPNeRF


CODE_SAVE_ITERS_ = [0, 5, 10, 20, 50, 100]
POSE_FW_ITERS_ = [0, 1, 2] + [i for i in range(5, 100, 5)]

WLH_MEAN = np.array([1.9446588, 4.641784, 1.7103361])
WLH_STD = np.array([0.1611075, 0.3961748, 0.20885137])

box_c = np.array([1, 144 / 255, 30 / 255]).astype(np.float64)
vis_im_sz = 128
puttext_ratio = vis_im_sz/128


class OptimizerNuScenes:
    def __init__(self, gpu, nusc_dataset, hpams, hpams_pose_refiner=None, hpams_pose_regressor=None,
                 model_epoch=None, code_level=0,
                 opt_pose=False, ada_posecode=0, reg_iters=3, opt_multiview=False,
                 pred_wlh=0, cross_eval_folder=None,
                 save_postfix='_nuscenes', num_workers=0, shuffle=False, save_freq=100, vis=0):
        super().__init__()
        self.hpams = hpams
        self.hpams_pose_refiner = hpams_pose_refiner
        self.hpams_pose_regressor = hpams_pose_regressor
        self.use_bn = True
        if 'norm_layer_type' in self.hpams['net_hyperparams'].keys() and \
                self.hpams['net_hyperparams']['norm_layer_type'] != 'BatchNorm2d':
            self.use_bn = False
        self.model_epoch = model_epoch
        self.code_level = code_level
        self.opt_pose = opt_pose
        self.ada_posecode = ada_posecode
        self.reg_iters = reg_iters
        self.opt_multiview = opt_multiview
        self.pred_wlh = pred_wlh
        self.cross_eval_folder = cross_eval_folder
        self.save_freq = save_freq
        self.vis = vis
        self.save_postfix = save_postfix
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model(self.model_epoch)
        if self.cross_eval_folder is None:
            self.make_save_img_dir()
            print('we are going to save at ', self.save_dir)
            self.cross_eval_folder = self.save_dir
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

        # initialize shapecode, texturecode, poses to optimize
        self.optimized_shapecodes = {}
        self.optimized_texturecodes = {}
        self.optimized_poses = {}

        # optimized poses are used for cross-view evaluation
        # poses are saved per img (per time step, per camera)
        latent_dim = self.hpams['net_hyperparams']['latent_dim']
        n_code = len(CODE_SAVE_ITERS_)
        for ii, sample in enumerate(self.nusc_dataset.all_valid_samples):
            (anntoken, cam) = sample
            if anntoken not in self.optimized_shapecodes.keys():
                self.optimized_poses[anntoken] = {}
            if cam not in self.optimized_poses[anntoken].keys():
                self.optimized_poses[anntoken][cam] = torch.zeros((3, 4), dtype=torch.float32)

        if self.code_level == 0:
            # codes are saved per ins (cross time, cross camera, do not cross scene)
            for ii, instoken in enumerate(self.nusc_dataset.anntokens_per_ins.keys()):
                if instoken not in self.optimized_shapecodes.keys():
                    self.optimized_shapecodes[instoken] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
                    self.optimized_texturecodes[instoken] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
        elif self.code_level == 1:
            # codes are saved per ann (per time step, may involve multiple cameras)
            for ii, anntoken in enumerate(self.nusc_dataset.instoken_per_ann.keys()):
                if anntoken not in self.optimized_shapecodes.keys():
                    self.optimized_shapecodes[anntoken] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
                    self.optimized_texturecodes[anntoken] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
        elif self.code_level == 2:
            # codes are saved per img (per time step, per camera)
            for ii, sample in enumerate(self.nusc_dataset.all_valid_samples):
                (anntoken, cam) = sample
                if anntoken not in self.optimized_shapecodes.keys():
                    self.optimized_shapecodes[anntoken] = {}
                    self.optimized_texturecodes[anntoken] = {}
                    self.optimized_poses[anntoken] = {}
                if cam not in self.optimized_shapecodes[anntoken].keys():
                    self.optimized_shapecodes[anntoken][cam] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
                    self.optimized_texturecodes[anntoken][cam] = torch.zeros((n_code, latent_dim), dtype=torch.float32)
                    self.optimized_poses[anntoken][cam] = torch.zeros((n_code, 3, 4), dtype=torch.float32)

        else:
            print('Error: invalid code_level input, has to be 0/1/2.')

        # TODO: should all record per ann per camera
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
            if self.opt_multiview:
                self.optimize_objs_multi_anns_w_pose()
            else:
                if self.hpams['arch'] == 'supnerf':
                    self.optimize_objs_w_pose_unified()
                else:
                    self.optimize_objs_w_pose()
        else:
            if self.opt_multiview:
                self.optimize_objs_multi_anns(opt_model=False, slack_tex=True)
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
            cam_poses = batch_data['cam_poses']
            wlh_batch = batch_data['wlh']
    
            tgt_img, tgt_cam, mask_occ, roi, K = \
                imgs[0], cam_poses[0], masks_occ[0], rois[0], cam_intrinsics[0]
    
            instoken, anntoken, cam_id = batch_data['instoken'][0], batch_data['anntoken'][0], batch_data['cam_ids'][0]
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
            log_idx = f'{anntoken}_{cam_id}'
    
            H, W = tgt_img.shape[0:2]
            roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
    
            # crop tgt img to roi
            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
            vis_img = tgt_img.clone()
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
                    self.optimized_shapecodes[anntoken][cam_id][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[anntoken][cam_id][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                loss_per_img = []
    
                rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(self.model, self.device,
                                                                                           tgt_img, mask_occ, tgt_cam,
                                                                                           obj_diag, cam_intrinsics[0],
                                                                                           roi,
                                                                                           self.hpams['n_samples'],
                                                                                           shapecode, texturecode,
                                                                                           self.hpams[
                                                                                               'shapenet_obj_cood'],
                                                                                           self.hpams['sym_aug'],
                                                                                           im_sz=self.hpams[
                                                                                               'render_im_sz'],
                                                                                           n_rays=None
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
                    self.output_single_view_vis(vis_img, mask_occ, tgt_cam.detach().cpu(), obj_diag, K, roi,
                                                shapecode, texturecode, obj_sz, log_idx)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    self.set_optimizers(shapecode, texturecode)
    
            # Save the optimized codes
            self.optimized_shapecodes[anntoken][cam_id][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[anntoken][cam_id][-1] = texturecode.detach().cpu()
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
            cam_poses = batch_data['cam_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses = batch_data['obj_poses']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            wlh_batch = batch_data['wlh']

            tgt_img, tgt_cam, pred_pose, mask_occ, roi, K = \
                imgs[0], cam_poses[0], obj_poses_w_err[0], masks_occ[0], \
                rois[0], cam_intrinsics[0]

            if self.hpams['optimize']['opt_cam_pose']:
                pred_pose = cam_poses_w_err[0]

            instoken, anntoken, cam_id = batch_data['instoken'][0], batch_data['anntoken'][0], batch_data['cam_ids'][0]
            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
            log_idx = f'{anntoken}_{cam_id}'

            H, W = tgt_img.shape[0:2]
            roi_new2 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)

            # crop tgt img to roi
            tgt_img = tgt_img[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]]
            vis_img = tgt_img.clone()
            mask_occ = mask_occ[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)

            if 'autorf' in self.hpams['arch'] or 'supnerf' in self.hpams['arch']:
                # preprocess image and predict shapecode and texturecode
                if self.use_bn:
                    img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                else:
                    img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                if 'supnerf' in self.hpams['arch']:
                    shapecode, texturecode, _, _ = self.model.encode_img(img_in.to(self.device))
                else:
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
            if self.opt_pose == 0:
                self.set_optimizers(shapecode, texturecode)
            else:
                self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[anntoken][cam_id][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[anntoken][cam_id][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                loss_per_img = []

                t2opt = trans_vec[0].unsqueeze(-1)
                if self.hpams['euler_rot']:
                    rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[0], 'XYZ')
                else:
                    rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[0])

                if not self.hpams['optimize']['opt_cam_pose']:
                    rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
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
                                                                                           n_rays=None
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

                if self.hpams['obj_sz_reg']:
                    sz_reg_samples = generate_obj_sz_reg_samples(obj_sz, obj_diag, self.hpams['shapenet_obj_cood'], tau=0.05)
                    loss_obj_sz = self.loss_obj_sz(sz_reg_samples, shapecode, texturecode)
                    loss = loss + self.hpams['loss_obj_sz_coef'] * loss_obj_sz

                loss.backward()

                # save the rgb loss for psnr computation (only include object mask, random sampled for efficiency)
                mask_rgb = occ_pixels.clone()
                mask_rgb[occ_pixels < 0] = 0
                loss_rgb2 = torch.sum((rgb_rays - rgb_tgt) ** 2 * mask_rgb) / (torch.sum(mask_rgb) + 1e-9)
                loss_per_img.append(loss_rgb2.detach().item())
                psnr = self.log_eval_psnr(loss_per_img, [log_idx])

                # log pose err
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
                                                                       obj_diag, K, roi_new2, x_vec, y_vec,
                                                                       self.hpams['n_samples'],
                                                                       shapecode, texturecode,
                                                                       self.hpams['shapenet_obj_cood'],
                                                                       self.hpams['sym_aug'])
                depth_err = self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, log_idx)
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    self.output_single_view_vis(vis_img, mask_occ, cam2opt.detach().cpu(), obj_diag, K, roi_new2,
                                                shapecode, texturecode, obj_sz, log_idx,
                                                psnr, depth_err, errs_R, errs_T)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    if self.opt_pose == 0:
                        self.set_optimizers(shapecode, texturecode)
                    else:
                        self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            # Save the optimized codes
            self.optimized_shapecodes[anntoken][cam_id][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[anntoken][cam_id][-1] = texturecode.detach().cpu()
            if batch_idx % self.save_freq == 0 or batch_idx == (len(self.dataloader) - 1):
                print(f'save result at batch {batch_idx}')
                self.save_opts_w_pose(batch_idx)

    def fw_pose_update(self,
                       im_feat_batch,
                       src_pose_batch,
                       wlh_batch,
                       roi_batch,
                       K_batch,
                       K_inv_batch,
                       iters=3,
                       start_wt_est_pose=False,
                       pred_uv_batch_direct=None
                       ):
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
            # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
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

        # regress delta_pose may be better than delta uv which may not consistent which is more dependant on ransac
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
            the pose refine process is unified with the NeRF optimization
            Optimize on each annotation frame independently
            Use additional pose refiner network to approach true pose iteratively
            If only to regress pose, the nerf model is only for the rendering and visualization purpose
        """

        # Per object
        for batch_idx, batch_data in enumerate(self.dataloader):
            # t0 = time.time()
            imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            obj_poses = batch_data['obj_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            wlh_batch = batch_data['wlh']
            K_inv_batch = torch.linalg.inv(cam_intrinsics)

            tgt_img, tgt_cam, pred_pose, mask_occ, roi, K = \
                imgs[0], cam_poses[0], obj_poses_w_err[0], masks_occ[0], \
                rois[0], cam_intrinsics[0]

            instoken, anntoken, cam_id = batch_data['instoken'][0], batch_data['anntoken'][0], batch_data['cam_ids'][0]
            log_idx = f'{anntoken}_{cam_id}'
            print(f'num obj: {batch_idx}/{len(self.dataloader)} - {log_idx}')

            obj_sz = wlh_batch[0].numpy()
            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

            H, W = tgt_img.shape[0:2]
            # roi process may differs for dl_update input image and nerf render setups
            # TODO: when training all used square, should this be simplified
            roi_new1 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=False)

            # crop tgt img to roi
            tgt_img = tgt_img[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]]
            mask_occ = mask_occ[roi_new1[1]: roi_new1[3], roi_new1[0]: roi_new1[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            tgt_img = tgt_img * (mask_occ > 0)
            tgt_img = tgt_img + (mask_occ <= 0)

            # preprocess image and
            img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
            # t1 = time.time()
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
            # t2 = time.time()
            # print(f'image coding time {t2-t1}s')

            #  run pose refiner before nerf optimization (TEMP not include in the overall iterations)
            with torch.no_grad():
                pose_per_iter = self.fw_pose_update(
                    posecode,
                    pred_pose.unsqueeze(0).to(self.device),
                    wlh_batch.to(self.device),
                    roi_new1.unsqueeze(0).to(self.device),
                    cam_intrinsics.to(self.device),
                    K_inv_batch.to(self.device),
                    iters=self.reg_iters,
                    start_wt_est_pose=self.opt_pose == 2,
                    pred_uv_batch_direct=pred_uv_batch_direct)
            # t3 = time.time()
            # print(f'pose estimation time {t3-t2}s')
            pose_per_iter = torch.from_numpy(np.array(pose_per_iter)).to(self.device)
            pred_pose = pose_per_iter[-1]
            # TODO: can check pred_pose, if too bad can trigger the dl pose update again from a different rand pose
            pred_up_axis = pred_pose[:, 2:3].cpu().detach().numpy()
            gt_up_axis = np.array([0, -1, 0]).reshape((3, 1))
            up_angle_diff = np.arccos(pred_up_axis.T @ gt_up_axis)[0, 0]
            if np.abs(up_angle_diff) > np.pi / 4:
                print('Found out-of-distribution pose')

            # set pose parameters
            rot_mat_vec = pred_pose[:3, :3].unsqueeze(0)
            Z_init = pred_pose[2, 3].clone().to(self.device)
            trans_vec = pred_pose[:3, 3].unsqueeze(0).to(self.device).detach().requires_grad_()
            rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimization
            self.nopts = 0
            if self.opt_pose == 0:
                self.set_optimizers(shapecode, texturecode)
            else:
                self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[anntoken][cam_id][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[anntoken][cam_id][code_i] = texturecode.detach().cpu()

                self.opts.zero_grad()
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
                    self.optimized_poses[anntoken][cam_id][code_i] = torch.cat((rot_mat2opt, t2opt), dim=-1).detach().cpu()

                if not self.hpams['optimize']['opt_cam_pose']:
                    rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
                    t2opt = -rot_mat2opt @ t2opt

                cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                # TODO: A different roi and crop for nerf (Can be removed when new model is trained for new infer)
                tgt_img = imgs[0]
                mask_occ = masks_occ[0]
                H, W = tgt_img.shape[0:2]
                roi_new2 = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)

                # crop tgt img to roi
                tgt_img = tgt_img[roi_new2[1]: roi_new2[3], roi_new2[0]: roi_new2[2]]
                vis_img = tgt_img.clone()
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
                                                                                           n_rays=None
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
                                                                       obj_diag, K, roi_new2, x_vec, y_vec,
                                                                       self.hpams['n_samples'],
                                                                       shapecode, texturecode,
                                                                       self.hpams['shapenet_obj_cood'],
                                                                       self.hpams['sym_aug'])
                depth_err = self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, log_idx)

                # The first a few is just for evaluation purpose to render rays, do not update codes or pose
                if self.nopts > self.reg_iters:
                    self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts <= self.reg_iters or
                                                        self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    self.output_single_view_vis(vis_img, mask_occ, cam2opt.detach().cpu(), obj_diag, K, roi_new2,
                                                shapecode, texturecode, obj_sz, log_idx,
                                                psnr, depth_err, errs_R, errs_T)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    if self.opt_pose == 0:
                        self.set_optimizers(shapecode, texturecode)
                    else:
                        self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            # t4 = time.time()
            # print(f'NeRF time {t4 - t3}s')
            # print(f'overall time per sample {t4 - t0}s')

            # Save the optimized codes
            self.optimized_shapecodes[anntoken][cam_id][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[anntoken][cam_id][-1] = texturecode.detach().cpu()
            self.optimized_poses[anntoken][cam_id][-1] = pred_obj_poses[0].detach().cpu()
            if batch_idx % self.save_freq == 0 or batch_idx == (len(self.dataloader) - 1):
                print(f'save result at batch {batch_idx}')
                self.save_opts_w_pose(batch_idx)

    def optimize_objs_multi_anns(self, opt_model=False, slack_tex=False):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """
    
        if slack_tex:
            # initial color adjustment saved per img (per time step, per camera)
            self.optimized_texture_res = {}
            latent_dim = self.hpams['net_hyperparams']['latent_dim']
            n_code = len(CODE_SAVE_ITERS_)
            for ii, sample in enumerate(self.nusc_dataset.all_valid_samples):
                (anntoken, cam) = sample
                if anntoken not in self.optimized_texture_res.keys():
                    self.optimized_texture_res[anntoken] = {}
                if cam not in self.optimized_texture_res[anntoken].keys():
                    self.optimized_texture_res[anntoken][cam] = torch.nn.Embedding(1, latent_dim)
                    self.optimized_texture_res[anntoken][cam].weight.data.fill_(0)
    
        instokens = self.nusc_dataset.anntokens_per_ins.keys()
        # Optimize per object instance
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')
    
            batch_data = self.nusc_dataset.get_ins_samples(instoken)
    
            tgt_imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            tgt_poses = batch_data['cam_poses']
            anntokens = batch_data['anntokens']
            wlh_batch = batch_data['wlh']
    
            if len(tgt_imgs) == 0:
                continue
    
            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
    
            if 'autorf' in self.hpams['arch']:
                # compute the mean shapecode and texturecode from different views
                shapecode_list, texturecode_list = [], []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, mask_occ, roi = tgt_imgs[num], masks_occ[num], rois[num]
                    # crop tgt img to roi
                    roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ <= 0)
    
                    # preprocess image and predict shapecode and texturecode
                    if self.use_bn:
                        img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                    else:
                        img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                    shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                    shapecode_list.append(shapecode)
                    texturecode_list.append(texturecode)
                shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
                texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')
    
            # Set up optimizer for the current object instance
            self.nopts = 0
            self.set_optimizers(shapecode, texturecode)
            if opt_model:
                model2opt = copy.deepcopy(self.model)
                self.opts.add_param_group({'params': model2opt.parameters(), 'lr': 0.001})
            else:
                model2opt = self.model
            if slack_tex:
                # add slack parameters for each view
                for num in range(0, tgt_imgs.shape[0]):
                    anntoken = batch_data['anntokens'][num]
                    cam_id = batch_data['cam_ids'][num]
                    texture_res = self.optimized_texture_res[anntoken][cam_id]
                    self.opts.add_param_group({'params': texture_res.parameters(), 'lr': self.hpams['optimize']['lr_texture']})
    
            # Start optimization
            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[instoken][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[instoken][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                # gt_imgs = []
                gt_masks_occ = []
                gt_depth_maps = []
                loss_per_img = []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], \
                                                          cam_intrinsics[num]
                    if slack_tex:
                        anntoken = batch_data['anntokens'][num]
                        cam_id = batch_data['cam_ids'][num]
                        texture_res = self.optimized_texture_res[anntoken][cam_id].to(self.device)
                        texturecode_ii = texturecode + texture_res.weight
                    else:
                        texturecode_ii = texturecode
    
                    # obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                    obj_sz = wlh_batch[num].numpy()
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
    
                    # crop tgt img to roi
                    roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ <= 0)
    
                    # render ray values and prepare target rays
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(model2opt, self.device,
                                                                                               tgt_img, mask_occ,
                                                                                               tgt_pose,
                                                                                               obj_diag,
                                                                                               K,
                                                                                               roi,
                                                                                               self.hpams['n_samples'],
                                                                                               shapecode, texturecode_ii,
                                                                                               self.hpams[
                                                                                                   'shapenet_obj_cood'],
                                                                                               self.hpams['sym_aug'],
                                                                                               im_sz=self.hpams[
                                                                                                   'render_im_sz'],
                                                                                               n_rays=None
                                                                                               # im_sz=64,
                                                                                               # n_rays=self.hpams['n_rays']
                                                                                               )
    
                    # Compute losses
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
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
    
                    # Different roi sizes are dealt in save_image later
                    # gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    gt_depth_maps.append(batch_data['depth_maps'][num, roi[1]:roi[3], roi[0]:roi[2]].numpy())
    
                    # log depth error every iter only rendering the lidar pixels
                    gt_depth_map = batch_data['depth_maps'][num, roi[1]:roi[3], roi[0]:roi[2]].numpy()
                    y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                    gt_depth_vec = gt_depth_map[y_vec, x_vec]
                    with torch.no_grad():
                        _, depth_pred_vec, _, _, _ = render_rays_specified(model2opt, self.device,
                                                                           tgt_img, mask_occ, tgt_pose,
                                                                           obj_diag, K, roi, x_vec, y_vec,
                                                                           self.hpams['n_samples'],
                                                                           shapecode, texturecode_ii,
                                                                           self.hpams['shapenet_obj_cood'],
                                                                           self.hpams['sym_aug'])
                    self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, anntokens[num])
                # if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                self.log_eval_psnr(loss_per_img, anntokens)
                self.opts.step()
    
                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    # generate the full images
                    generated_imgs = []
                    generated_depth_maps = []
                    gt_imgs = []
                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[
                                num], cam_intrinsics[num]
                            if slack_tex:
                                anntoken = batch_data['anntokens'][num]
                                cam_id = batch_data['cam_ids'][num]
                                texture_res = self.optimized_texture_res[anntoken][cam_id].to(self.device)
                                texturecode_ii = texturecode + texture_res.weight
                            else:
                                texturecode_ii = texturecode
    
                            # crop tgt img to roi
                            roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
                            tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
    
                            # obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            obj_sz = wlh_batch[num].numpy()
                            obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
    
                            vis_im_sz = 128
                            rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(model2opt, self.device,
                                                                                    tgt_img, mask_occ, tgt_pose,
                                                                                    obj_diag, K,
                                                                                    roi, self.hpams['n_samples'],
                                                                                    shapecode, texturecode_ii,
                                                                                    self.hpams['shapenet_obj_cood'],
                                                                                    self.hpams['sym_aug'],
                                                                                    im_sz=vis_im_sz, n_rays=None)
                            generated_img = rgb_rays.view((vis_im_sz, vis_im_sz, 3))
                            generated_depth = depth_rays.view((vis_im_sz, vis_im_sz))
                            generated_imgs.append(generated_img.cpu())
                            generated_depth_maps.append(generated_depth.cpu())
                            gt_img = tgt_img.cpu().unsqueeze(0)
                            gt_img = gt_img.permute((0, 3, 1, 2))
                            gt_img = Resize((vis_im_sz, vis_im_sz))(gt_img).permute((0, 2, 3, 1))[0]
                            # overlay mask for visualization
                            mask_occ = mask_occ.cpu().unsqueeze(0)
                            mask_occ = mask_occ.permute((0, 3, 1, 2))
                            mask_occ = Resize((vis_im_sz, vis_im_sz))(mask_occ).permute((0, 2, 3, 1))[0]
                            gt_img = gt_img * 0.75 + mask_occ * 0.25
                            gt_imgs.append(gt_img)
                        # self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)
                        self.save_img3(generated_imgs,
                                       generated_depth_maps,
                                       gt_imgs, instoken, self.nopts)
    
                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[0])['size']
                            virtual_imgs = render_virtual_imgs(model2opt, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)
    
                self.nopts += 1
                # if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                #     self.set_optimizers(shapecode, texturecode)
    
            # Save the optimized codes
            self.optimized_shapecodes[instoken][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken][-1] = texturecode.detach().cpu()
            # self.log_eval_depth(generated_depth_maps, gt_depth_maps, gt_masks_occ, anntokens)
            if obj_idx % self.save_freq == 0 or obj_idx == (len(instokens)-1):
                print(f'Save result at completing {obj_idx} instances.')
                self.save_opts(obj_idx)

    def optimize_objs_multi_anns_w_pose(self):
        """
            optimize multiple annotations for the same instance in a singe iteration
        """

        instokens = self.nusc_dataset.anntokens_per_ins.keys()
        # Per object
        for obj_idx, instoken in enumerate(instokens):
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}')

            batch_data = self.nusc_dataset.get_ins_samples(instoken)

            tgt_imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            cam_poses = batch_data['cam_poses']
            cam_poses_w_err = batch_data['cam_poses_w_err']
            obj_poses = batch_data['obj_poses']
            obj_poses_w_err = batch_data['obj_poses_w_err']
            anntokens = batch_data['anntokens']

            if self.hpams['optimize']['opt_cam_pose']:
                pred_poses = cam_poses_w_err
            else:
                pred_poses = obj_poses_w_err

            if len(tgt_imgs) == 0:
                continue

            print(f'    num views: {tgt_imgs.shape[0]}')
            H, W = tgt_imgs.shape[1:3]
            rois[..., 0:2] -= self.hpams['roi_margin']
            rois[..., 2:4] += self.hpams['roi_margin']
            rois[..., 0:2] = torch.maximum(rois[..., 0:2], torch.as_tensor(0))
            rois[..., 2] = torch.minimum(rois[..., 2], torch.as_tensor(W - 1))
            rois[..., 3] = torch.minimum(rois[..., 3], torch.as_tensor(H - 1))

            if 'autorf' in self.hpams['arch']:
                # compute the mean shapecode and texturecode from different views
                shapecode_list, texturecode_list = [], []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, mask_occ, roi = tgt_imgs[num], masks_occ[num], rois[num]

                    # crop tgt img to roi
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ <= 0)

                    # preprocess image and predict shapecode and texturecode
                    if self.use_bn:
                        img_in = preprocess_img_square(tgt_img, self.hpams['in_img_sz'])
                    else:
                        img_in = preprocess_img_keepratio(tgt_img, self.hpams['max_img_sz'])
                    shapecode, texturecode = self.model.encode_img(img_in.to(self.device))
                    shapecode_list.append(shapecode)
                    texturecode_list.append(texturecode)
                shapecode = torch.mean(torch.cat(shapecode_list), dim=0, keepdim=True).detach().requires_grad_()
                texturecode = torch.mean(torch.cat(texturecode_list), dim=0, keepdim=True).detach().requires_grad_()
            elif self.hpams['arch'] == 'codenerf':
                shapecode = self.mean_shape.clone().to(self.device).detach().requires_grad_()
                texturecode = self.mean_texture.clone().to(self.device).detach().requires_grad_()
            else:
                shapecode = None
                texturecode = None
                print('ERROR: No valid network architecture is declared in config file!')

            # set pose parameters
            rot_mat_vec = pred_poses[:, :3, :3]
            trans_vec = pred_poses[:, :3, 3].to(self.device).detach().requires_grad_()
            if self.hpams['euler_rot']:
                rot_vec = rot_trans.matrix_to_euler_angles(rot_mat_vec, 'XYZ').to(self.device).detach().requires_grad_()
            else:
                rot_vec = rot_trans.matrix_to_axis_angle(rot_mat_vec).to(self.device).detach().requires_grad_()

            # Optimize
            self.nopts = 0
            if self.opt_pose == 0:
                self.set_optimizers(shapecode, texturecode)
            else:
                self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)

            est_poses = torch.zeros((tgt_imgs.shape[0], 3, 4), dtype=torch.float32)
            pred_obj_poses = torch.zeros((tgt_imgs.shape[0], 3, 4), dtype=torch.float32)
            while self.nopts < self.hpams['optimize']['num_opts']:
                if self.nopts in CODE_SAVE_ITERS_:
                    code_i = CODE_SAVE_ITERS_.index(self.nopts)
                    self.optimized_shapecodes[instoken][code_i] = shapecode.detach().cpu()
                    self.optimized_texturecodes[instoken][code_i] = texturecode.detach().cpu()
                self.opts.zero_grad()
                t1 = time.time()
                gt_imgs = []
                gt_masks_occ = []
                gt_depth_maps = []
                loss_per_img = []
                for num in range(0, tgt_imgs.shape[0]):
                    tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], cam_poses[num], masks_occ[num], rois[num], \
                                                          cam_intrinsics[num]

                    t2opt = trans_vec[num].unsqueeze(-1)
                    if self.hpams['euler_rot']:
                        rot_mat2opt = rot_trans.euler_angles_to_matrix(rot_vec[num], 'XYZ')
                    else:
                        rot_mat2opt = rot_trans.axis_angle_to_matrix(rot_vec[num])

                    if not self.hpams['optimize']['opt_cam_pose']:
                        rot_mat2opt = torch.transpose(rot_mat2opt, dim0=-2, dim1=-1)
                        t2opt = -rot_mat2opt @ t2opt

                    cam2opt = torch.cat((rot_mat2opt, t2opt), dim=-1)

                    obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                    # crop tgt img to roi
                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                    # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                    tgt_img = tgt_img * (mask_occ > 0)
                    tgt_img = tgt_img + (mask_occ <= 0)

                    # render ray values and prepare target rays
                    rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays(self.model, self.device,
                                                                                            tgt_img, mask_occ, cam2opt,
                                                                                            obj_diag, K, roi,
                                                                                            self.hpams['n_rays'],
                                                                                            self.hpams['n_samples'],
                                                                                            shapecode, texturecode,
                                                                                            self.hpams['shapenet_obj_cood'],
                                                                                            self.hpams['sym_aug'])

                    # Compute losses
                    loss_rgb = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)
                    # Occupancy loss
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

                    # Different roi sizes are dealt in save_image later
                    gt_imgs.append(tgt_imgs[num, roi[1]:roi[3], roi[0]:roi[2]])  # only include the roi area
                    gt_masks_occ.append(masks_occ[num, roi[1]:roi[3], roi[0]:roi[2]])
                    gt_depth_maps.append(batch_data['depth_maps'][num, roi[1]:roi[3], roi[0]:roi[2]].numpy())
                    # self.optimized_pose_flag[anntokens[num]] = 1
                    est_poses[num] = cam2opt.detach().cpu()

                    pred_obj_R = cam2opt[:, :3].detach().cpu().transpose(-2, -1)
                    pred_obj_T = -pred_obj_R @ (cam2opt[:, 3:].detach().cpu())
                    pred_obj_poses[num] = torch.cat([pred_obj_R, pred_obj_T], dim=-1)

                    # log depth error every iter only rendering the lidar pixels
                    gt_depth_map = batch_data['depth_maps'][num, roi[1]:roi[3], roi[0]:roi[2]].numpy()
                    y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                    gt_depth_vec = gt_depth_map[y_vec, x_vec]
                    with torch.no_grad():
                        _, depth_pred_vec, _, _, _ = render_rays_specified(self.model, self.device,
                                                                           tgt_img, mask_occ, cam2opt,
                                                                           obj_diag, K, roi, x_vec, y_vec,
                                                                           self.hpams['n_samples'],
                                                                           shapecode, texturecode,
                                                                           self.hpams['shapenet_obj_cood'],
                                                                           self.hpams['sym_aug'])
                    self.log_eval_depth_v2(depth_pred_vec.cpu().numpy(), gt_depth_vec, anntokens[num])
                errs_R, errs_T = self.log_eval_pose(pred_obj_poses, obj_poses, anntokens)
                self.log_eval_psnr(loss_per_img, anntokens)
                self.opts.step()

                # Just render the cropped region instead to save computation on the visualization
                # ATTENTION: the optimizing parameters are updated, but intermediate variables are not
                if self.vis == 2 or (self.vis == 1 and (self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1))):
                    # generate the full images
                    generated_imgs = []
                    generated_depth_maps = []

                    with torch.no_grad():
                        for num in range(0, tgt_imgs.shape[0]):
                            tgt_pose, roi, K = cam_poses[num], rois[num], cam_intrinsics[num]
                            cam2opt = est_poses[num]
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                            # render full image
                            generated_img, generated_depth = render_full_img(self.model, self.device, cam2opt, obj_sz,
                                                                             K, roi, self.hpams['n_samples'],
                                                                             shapecode, texturecode,
                                                                             self.hpams['shapenet_obj_cood'],
                                                                             out_depth=True)
                            # mark pose error on the image
                            err_str = 'RE: {:.3f}, TE: {:.3f}'.format(errs_R[num], errs_T[num])
                            generated_img = cv2.putText(generated_img.cpu().numpy(), err_str, (5, 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, .3, (1, 0, 0), thickness=2)
                            generated_imgs.append(torch.from_numpy(generated_img))
                            generated_depth_maps.append(generated_depth.cpu().numpy())

                            # save the last pose for later evaluation
                            if self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                                est_poses[num] = cam2opt.detach().cpu()
                        self.save_img(generated_imgs, gt_imgs, gt_masks_occ, instoken, self.nopts)

                        # save virtual views at the beginning and the end
                        if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
                            obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[0])['size']
                            virtual_imgs = render_virtual_imgs(self.model, self.device, obj_sz, cam_intrinsics[0],
                                                               self.hpams['n_samples'], shapecode, texturecode,
                                                               self.hpams['shapenet_obj_cood'])
                            self.save_virtual_img(virtual_imgs, instoken, self.nopts)
                self.nopts += 1
                if self.nopts % self.hpams['optimize']['lr_half_interval'] == 0:
                    if self.opt_pose == 0:
                        self.set_optimizers(shapecode, texturecode)
                    else:
                        self.set_optimizers_w_poses(shapecode, texturecode, rot_vec, trans_vec)
            # Save the optimized codes
            self.optimized_shapecodes[instoken][-1] = shapecode.detach().cpu()
            self.optimized_texturecodes[instoken][-1] = texturecode.detach().cpu()
            # self.log_eval_depth(generated_depth_maps, gt_depth_maps, gt_masks_occ, anntokens)
            if obj_idx % self.save_freq == 0 or obj_idx == (len(instokens)-1):
                print(f'Save result at completing {obj_idx} instances.')
                self.save_opts_w_pose(obj_idx)

    def eval_cross_view(self, vis_iter=None):
        """
            Conduct cross-view evaluation based on previous saved optimization results
        """
        print('Cross-view evaluation of nerf reconstruction based on previous saved codes ...')

        if self.cross_eval_folder is not None:
            self.save_dir = self.cross_eval_folder

        # if self.opt_pose:
        result_file = os.path.join(self.save_dir, 'codes+poses.pth')
        # else:
        #     result_file = os.path.join(self.save_dir, 'codes.pth')

        saved_result = torch.load(result_file, map_location=torch.device('cpu'))

        self.optimized_shapecodes = saved_result['optimized_shapecodes']
        self.optimized_texturecodes = saved_result['optimized_texturecodes']
        self.lidar_pts_cnt = saved_result['lidar_pts_cnt']

        instokens = self.nusc_dataset.anntokens_per_ins.keys()
        psnr_eval_mat_per_ins = {}
        depth_eval_mat_per_ins = {}
        cnt_lidar_pts_per_ins = {}

        # Per object
        for obj_idx, instoken in enumerate(instokens):
            batch_data = self.nusc_dataset.get_ins_samples(instoken)

            tgt_imgs = batch_data['imgs']
            masks_occ = batch_data['masks_occ']
            rois = batch_data['rois']
            cam_intrinsics = batch_data['cam_intrinsics']
            tgt_poses = batch_data['cam_poses']
            anntokens = batch_data['anntokens']
            cam_ids = batch_data['cam_ids']
            H, W = tgt_imgs.shape[1:3]

            num_imgs = tgt_imgs.shape[0]
            if num_imgs < 2:
                continue
            print(f'num obj: {obj_idx}/{len(instokens)}, instoken: {instoken}, num samples: {num_imgs}')

            for cid in range(0, len(CODE_SAVE_ITERS_)):
                psnr_eval_mat = np.zeros((num_imgs, num_imgs))
                depth_eval_mat = np.zeros((num_imgs, num_imgs))
                # lidar_cnt_vec = []
                # For each code, generate rgb and depth for all the other views and evaluate
                empty_rows = []
                for ii in range(0, num_imgs):
                    if anntokens[ii] not in self.optimized_shapecodes.keys():
                        empty_rows.append(ii)
                        continue
                    shapecode = self.optimized_shapecodes[anntokens[ii]][cam_ids[ii]][cid:cid+1].to(self.device)
                    texturecode = self.optimized_texturecodes[anntokens[ii]][cam_ids[ii]][cid:cid+1].to(self.device)
                    # lidar_cnt_vec.append(self.lidar_pts_cnt[anntokens[ii]])
                    for num in range(0, num_imgs):
                        tgt_img, tgt_pose, mask_occ, roi, K = tgt_imgs[num], tgt_poses[num], masks_occ[num], rois[num], \
                                                              cam_intrinsics[num]

                        obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntokens[num])['size']
                        obj_diag = np.linalg.norm(obj_sz).astype(np.float32)

                        # crop tgt img to roi
                        roi = roi_process(roi, H, W, self.hpams['roi_margin'], sq_pad=True)
                        tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                        mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
                        # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
                        tgt_img = tgt_img * (mask_occ > 0)
                        tgt_img = tgt_img + (mask_occ <= 0)

                        with torch.no_grad():
                            rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels = render_rays_v2(
                                self.model, self.device,
                                tgt_img, mask_occ, tgt_pose, obj_diag, K, roi,
                                self.hpams['n_samples'], shapecode, texturecode,
                                self.hpams['shapenet_obj_cood'], self.hpams['sym_aug'],
                                im_sz=self.hpams['render_im_sz'], n_rays=None)
                        # Compute losses
                        # Critical to let rgb supervised on white background
                        loss_rgb2 = torch.sum((rgb_rays - rgb_tgt) ** 2 * torch.abs(occ_pixels)) / (
                                torch.sum(torch.abs(occ_pixels)) + 1e-9)

                        psnr = -10 * np.log(loss_rgb2.item()) / np.log(10)
                        psnr_eval_mat[ii, num] = psnr

                        # log depth error every iter only rendering the lidar pixels
                        gt_depth_map = batch_data['depth_maps'][num, roi[1]:roi[3], roi[0]:roi[2]].numpy()
                        y_vec, x_vec = np.where(np.logical_and(gt_depth_map > 0, mask_occ[:, :, 0].numpy() > 0))
                        gt_depth_vec = gt_depth_map[y_vec, x_vec]
                        with torch.no_grad():
                            _, depth_pred_vec, _, _, _ = render_rays_specified(self.model, self.device,
                                                                               tgt_img, mask_occ, tgt_pose,
                                                                               obj_diag, K, roi, x_vec, y_vec,
                                                                               self.hpams['n_samples'],
                                                                               shapecode, texturecode,
                                                                               self.hpams['shapenet_obj_cood'],
                                                                               self.hpams['sym_aug'])

                        depth_errs = abs(depth_pred_vec.cpu().numpy() - gt_depth_vec)
                        depth_err_mean = np.sum(depth_errs) / (len(gt_depth_vec) + 1e-8)
                        depth_eval_mat[ii, num] = depth_err_mean

                        if vis_iter is not None and CODE_SAVE_ITERS_[cid] == vis_iter:
                            vis_name = f'Iter_{vis_iter}_rec_{ii}_rend_{num}'
                            self.output_cross_view_vis(tgt_img, mask_occ, tgt_pose, obj_diag, K, roi,
                                                       shapecode, texturecode, obj_sz, 'crossview_'+ instoken, vis_name,
                                                       psnr, depth_err_mean)

                psnr_eval_mat = np.delete(psnr_eval_mat, empty_rows, 0)
                psnr_eval_mat = np.delete(psnr_eval_mat, empty_rows, 1)
                depth_eval_mat = np.delete(depth_eval_mat, empty_rows, 0)
                depth_eval_mat = np.delete(depth_eval_mat, empty_rows, 1)

                if instoken not in psnr_eval_mat_per_ins.keys():
                    psnr_eval_mat_per_ins[instoken] = [psnr_eval_mat]
                    depth_eval_mat_per_ins[instoken] = [depth_eval_mat]
                    # cnt_lidar_pts_per_ins[instoken] = [np.array(lidar_cnt_vec)]
                else:
                    psnr_eval_mat_per_ins[instoken].append(psnr_eval_mat)
                    depth_eval_mat_per_ins[instoken].append(depth_eval_mat)
                    # cnt_lidar_pts_per_ins[instoken].append(np.array(lidar_cnt_vec))
                # TODO: how to visualize in images? --> only the top a few?

        # save a separate file to cross_eval results
        # TODO: just save json to save space?
        cross_eval_result = {'psnr_eval_mat_per_ins': psnr_eval_mat_per_ins,
                             'depth_eval_mat_per_ins': depth_eval_mat_per_ins,
                             'cnt_lidar_pts_per_ins': cnt_lidar_pts_per_ins,
                             'CODE_SAVE_ITERS_': CODE_SAVE_ITERS_}
        torch.save(cross_eval_result, os.path.join(self.save_dir, 'cross_eval.pth'))
        print('Done cross-view evaluation of photometric and depth reconstruction.')

    def loss_obj_sz(self, sz_reg_samples, shapecode, texturecode):
        samples_out = np.concatenate((np.expand_dims(sz_reg_samples['X_planes_out'], 0),
                                    np.expand_dims(sz_reg_samples['Y_planes_out'], 0),
                                    np.expand_dims(sz_reg_samples['Z_planes_out'], 0)), axis=0)
        samples_out = torch.from_numpy(samples_out)
        samples_in = np.concatenate((np.expand_dims(sz_reg_samples['X_planes_in'], 0),
                                    np.expand_dims(sz_reg_samples['Y_planes_in'], 0),
                                    np.expand_dims(sz_reg_samples['Z_planes_in'], 0)), axis=0)
        samples_in = torch.from_numpy(samples_in)

        sigmas_out, _ = self.model(samples_out.to(self.device),
                                 torch.ones_like(samples_out).to(self.device),
                                 shapecode, texturecode)
        sigmas_in, _ = self.model(samples_in.to(self.device),
                                 torch.ones_like(samples_in).to(self.device),
                                 shapecode, texturecode)
        sigmas_out_max = torch.max(sigmas_out.squeeze(), dim=1).values
        sigmas_in_max = torch.max(sigmas_in.squeeze(), dim=1).values

        loss = torch.sum(sigmas_out_max ** 2) + \
               torch.sum((sigmas_in_max - torch.ones_like(sigmas_in_max))**2)
        return loss / 6

    def loss_sym(self, xyz, viewdir, sigmas, shapecode, texturecode):
        xyz_sym = torch.clone(xyz)
        viewdir_sym = torch.clone(viewdir)
        if self.hpams['shapenet_obj_cood']:
            xyz_sym[:, :, 0] *= (-1)
            viewdir_sym[:, :, 0] *= (-1)
        else:
            xyz_sym[:, :, 1] *= (-1)
            viewdir_sym[:, :, 1] *= (-1)
        sigmas_sym, rgbs_sym = self.model(xyz_sym.to(self.device),
                                          viewdir_sym.to(self.device),
                                          shapecode, texturecode)
        loss_sym = torch.mean((sigmas - sigmas_sym) ** 2)
        return loss_sym

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

    def output_single_view_vis(self, vis_img, mask_occ, cam_pose, obj_diag, K, roi, shapecode, texturecode,
                               wlh, log_idx, psnr=None, depth_err=None, errs_R=None, errs_T=None):
        with torch.no_grad():

            rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(self.model, self.device,
                                                                    vis_img, mask_occ, cam_pose,
                                                                    obj_diag, K,
                                                                    roi, self.hpams['n_samples'],
                                                                    shapecode, texturecode,
                                                                    self.hpams['shapenet_obj_cood'],
                                                                    self.hpams['sym_aug'],
                                                                    im_sz=vis_im_sz, n_rays=None)
            # TODO: add maks based on depth_rays to remove background artifacts
            generated_img = rgb_rays.view((vis_im_sz, vis_im_sz, 3))
            generated_depth = depth_rays.view((vis_im_sz, vis_im_sz))
            # colorized_depth = colorize(generated_depth, 0, 60)

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

            u_vec_new, v_vec_new = roi_coord_trans(pred_uv[0, :], pred_uv[1, :],
                                                   roi.numpy(), im_sz_tgt=vis_im_sz)
            pred_uv[0, :] = u_vec_new
            pred_uv[1, :] = v_vec_new
            vis_img = render_box(vis_img.cpu().numpy(), pred_uv, colors=(box_c, box_c, box_c))
            vis_img = torch.from_numpy(vis_img)
            # generated_img = render_box(generated_img.cpu().numpy(), pred_uv, colors=(box_c, box_c, box_c))
            # generated_img = torch.from_numpy(generated_img)

            self.save_img3([generated_img],
                           [generated_depth],
                           [vis_img], log_idx, self.nopts,
                           psnr, [depth_err], errs_R, errs_T)

            # # save virtual views at the beginning and the end
            # if self.nopts == 0 or self.nopts == (self.hpams['optimize']['num_opts'] - 1):
            #     virtual_imgs = render_virtual_imgs(self.model, self.device, wlh, K,
            #                                        self.hpams['n_samples'], shapecode, texturecode,
            #                                        self.hpams['shapenet_obj_cood'])
            #     self.save_virtual_img(virtual_imgs, log_idx, self.nopts)

    def output_cross_view_vis(self, vis_img, mask_occ, cam_pose, obj_diag, K, roi, shapecode, texturecode,
                              wlh, ins_id, vis_name, psnr=None, depth_err=None):
        with torch.no_grad():
            rgb_rays, depth_rays, _, _, occ_pixels = render_rays_v2(self.model, self.device,
                                                                    vis_img, mask_occ, cam_pose,
                                                                    obj_diag, K,
                                                                    roi, self.hpams['n_samples'],
                                                                    shapecode, texturecode,
                                                                    self.hpams['shapenet_obj_cood'],
                                                                    self.hpams['sym_aug'],
                                                                    im_sz=vis_im_sz, n_rays=None)
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

            u_vec_new, v_vec_new = roi_coord_trans(pred_uv[0, :], pred_uv[1, :],
                                                   roi.numpy(), im_sz_tgt=vis_im_sz)
            pred_uv[0, :] = u_vec_new
            pred_uv[1, :] = v_vec_new
            # generated_img = render_box(generated_img.cpu().numpy(), pred_uv, colors=(g_c, g_c, g_c))
            vis_img = render_box(vis_img.cpu().numpy(), pred_uv, colors=(box_c, box_c, box_c))

            self.save_img3([generated_img],
                           [generated_depth],
                           [torch.from_numpy(vis_img)], ins_id, None,
                           [psnr], [depth_err], cross_name=vis_name)

    def save_img(self, generated_imgs, gt_imgs, masks_occ, obj_id,  n_opts):
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
        imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(n_opts) + '.png'), ret)

    def save_img3(self, gen_rgb_imgs, gen_depth_imgs, gt_imgs, obj_id, n_opts, psnr=None, depth_err=None, rot_err=None, trans_err=None, cross_name=None):
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
            err_str1 = 'PSNR: {:.3f},  DE: {:.3f}'.format(
                psnr[0], depth_err[0])
            ret = cv2.putText(ret, err_str1, (int(5*puttext_ratio), int(10*puttext_ratio)), cv2.FONT_HERSHEY_SIMPLEX,
                              .35*puttext_ratio, (0, 0, 0), thickness=int(puttext_ratio))
        if rot_err is not None and trans_err is not None:
            err_str2 = 'RE: {:.3f},  TE: {:.3f}'.format(
                rot_err[0], trans_err[0])
            ret = cv2.putText(ret, err_str2, (int(5*puttext_ratio), int(21*puttext_ratio)), cv2.FONT_HERSHEY_SIMPLEX,
                              .35*puttext_ratio, (0, 0, 0), thickness=int(puttext_ratio))

        # err_str2 = 'RE: {:.2f}, TE: {:.2f}'.format(
        #     rot_err[0], trans_err[0])
        # ret = cv2.putText(ret, err_str2, (int(5 * 1.8), int(21 * 1.8)), cv2.FONT_HERSHEY_SIMPLEX,
        #                   .35 * 1.8, (0, 0, 0), thickness=2)

        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if cross_name is None:
            imageio.imwrite(os.path.join(save_img_dir, 'opt' + '{:03d}'.format(n_opts) + '.png'), ret)
        else:
            imageio.imwrite(os.path.join(save_img_dir, cross_name + '.png'), ret)

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

    # TODO: should the log consider different code levels?
    # TODO: the same ann can be optimized multiple times from different views
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

        errs_R, errs_T = calc_pose_err(est_poses, tgt_poses)

        for i, ann_token in enumerate(ann_tokens):
            if self.R_eval.get(ann_token) is None:
                self.R_eval[ann_token] = [errs_R[i]]
                self.T_eval[ann_token] = [errs_T[i]]
            else:
                self.R_eval[ann_token].append(errs_R[i])
                self.T_eval[ann_token].append(errs_T[i])
            if math.isnan(errs_T[i]) or math.isnan(errs_R[i]):
                print('FOUND NaN')
            if self.nopts == 0:
                print('   Initial RE: {:.3f}, TE: {:.3f}'.format(errs_R[i], errs_T[i]))
            if self.nopts == self.hpams['optimize']['num_opts'] - 1:
                print('   Final RE: {:.3f}, TE: {:.3f}'.format(errs_R[i], errs_T[i]))

        return errs_R, errs_T

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
