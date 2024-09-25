import numpy as np
import os
import time
import json
import math
import random
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch3d.transforms.rotation_conversions as rot_trans

from utils import image_float_to_uint8, render_full_img, prepare_pixel_samples, draw_boxes_train,\
    volume_rendering_batch, preprocess_img_square, normalize_by_roi, corners_of_box_batch, view_points_batch, roi_process, roi_resize
from model_supnerf import SUPNeRF


class ParallelModel(nn.Module):
    def __init__(self, model=None, hpams=None, im_enc_rate=1.0, pred_wlh: bool = False):
        super().__init__()
        self.model = model
        self.hpams = hpams
        self.im_enc_rate = im_enc_rate
        self.pred_wlh = pred_wlh
    
    def forward(self,
                img_in_batch,
                shapecode_batch,
                texturecode_batch,
                xyz_batch,
                viewdir_batch,
                z_vals_batch,
                rgb_tgt_batch,
                occ_pixels_batch,
                src_pose_batch,
                tgt_uv_batch,
                tgt_wlh_batch,
                roi_batch,
                K_batch,
                wlh_batch_aug,
                tgt_uv_batch_aug
                ):
        losses_all = {}
        loss_total = 0.

        """
            Common image encoding
        """
        # When encoder is triggered, codes will be used just once
        if self.pred_wlh:
            shapecode, texturecode, posecode, pred_uv_batch_direct, pred_wlh_batch = self.model.encode_img(img_in_batch)
            loss_wlh = ((pred_wlh_batch - tgt_wlh_batch) ** 2).mean()  # using sqrt can lead to NaN
            losses_all['loss_wlh'] = loss_wlh
            loss_total += self.hpams['loss_wlh_coef'] * losses_all['loss_wlh']
        else:
            shapecode, texturecode, posecode, pred_uv_batch_direct, _ = self.model.encode_img(img_in_batch)

        # Random trigger encoder is a way to mix autorf and codenerf mode in training
        enc_active = False
        if random.uniform(0, 1) < self.im_enc_rate:
            enc_active = True

        pred_uv_batch_direct = pred_uv_batch_direct.view((-1, 2, 8))
        # convert to original image frame
        dim_batch = torch.maximum(roi_batch[:, 2] - roi_batch[:, 0], roi_batch[:, 3] - roi_batch[:, 1])
        pred_uv_batch_direct *= (dim_batch.view((-1, 1, 1)) / 2)
        pred_uv_batch_direct[:, 0, :] += ((roi_batch[:, 0:1] + roi_batch[:, 2:3]) / 2)
        pred_uv_batch_direct[:, 1, :] += ((roi_batch[:, 1:2] + roi_batch[:, 3:4]) / 2)

        loss_uv = torch.sqrt(torch.sum((pred_uv_batch_direct[:, :2, :] - tgt_uv_batch) ** 2, dim=-2))
        losses_all['loss_pose_direct'] = loss_uv.mean()
        if enc_active:
            loss_total += self.hpams['loss_pose_coef'] * losses_all['loss_pose_direct']

        # TODO: any better way to apply this code consistency loss? --> only for those multiview sample?
        loss_code = torch.mean((shapecode - shapecode_batch)**2 + (texturecode - texturecode_batch)**2)
        losses_all['loss_code'] = loss_code
        if enc_active:
            # loss_code is expected to improve encoder, but unstable if fully encoder mode
            if self.im_enc_rate < 1.0:
                loss_total += self.hpams['loss_code_coef'] * losses_all['loss_code']
            shapecode_batch = (shapecode_batch + shapecode) / 2
            texturecode_batch = (texturecode_batch + texturecode) / 2

        """
            Pose stream (Need to BP with RGB loss to ensure stability of encoder training)
        """
        loss_1, pred_pose_batch1 = self.pose_regress(posecode,
                                                     src_pose_batch,
                                                     tgt_uv_batch_aug,
                                                     wlh_batch_aug,
                                                     roi_batch,
                                                     K_batch)

        loss_2, pred_pose_batch2 = self.pose_regress(posecode,
                                                     pred_pose_batch1,
                                                     tgt_uv_batch_aug,
                                                     wlh_batch_aug,
                                                     roi_batch,
                                                     K_batch)

        loss_3, pred_pose_batch3 = self.pose_regress(posecode,
                                                     pred_pose_batch2,
                                                     tgt_uv_batch_aug,
                                                     wlh_batch_aug,
                                                     roi_batch,
                                                     K_batch)

        losses_all['loss_pose_iter1'] = loss_1.mean()
        losses_all['loss_pose_iter2'] = loss_2.mean()
        losses_all['loss_pose_iter3'] = loss_3.mean()
        if enc_active:
            loss_total += self.hpams['loss_pose_coef'] * (
                    losses_all['loss_pose_iter1'] + losses_all['loss_pose_iter2'] + losses_all['loss_pose_iter3'])/3

        """
            Nerf subnetwork
        """
        sigmas, rgbs = self.model(xyz_batch.flatten(0, 1),
                                  viewdir_batch.flatten(0, 1),
                                  shapecode_batch,
                                  texturecode_batch)

        b_size = img_in_batch.shape[0]
        n, s, _ = sigmas.shape
        rgb_rays, depth_rays, acc_trans_rays = volume_rendering_batch(sigmas.view(b_size, int(n/b_size), s, -1),
                                                                      rgbs.view(b_size, int(n/b_size), s, -1),
                                                                      z_vals_batch)
        loss_rgb = torch.sum((rgb_rays - rgb_tgt_batch) ** 2 * torch.abs(occ_pixels_batch), dim=[-2, -1])/(
                torch.sum(torch.abs(occ_pixels_batch), dim=[-2, -1]) + 1e-9)
        losses_all['loss_rgb'] = loss_rgb.mean()

        psnr = -torch.tensor(10) * torch.log(loss_rgb.mean()) / torch.log(torch.tensor(10))
        losses_all['psnr'] = psnr.detach()

        # Occupancy loss
        loss_occ = torch.sum(
            torch.exp(-occ_pixels_batch * (0.5 - acc_trans_rays.unsqueeze(-1))) * torch.abs(occ_pixels_batch),
            dim=[-2, -1]) / (torch.sum(torch.abs(occ_pixels_batch), dim=[-2, -1]) + 1e-9)
        losses_all['loss_occ'] = loss_occ.mean()

        loss_reg = torch.norm(shapecode_batch, dim=-1) + torch.norm(texturecode_batch, dim=-1)
        losses_all['loss_reg'] = loss_reg.mean()

        loss_total += (losses_all['loss_rgb'] + self.hpams['loss_occ_coef'] * losses_all['loss_occ'])
        losses_all['loss_total'] = loss_total
        return losses_all, loss_total, shapecode_batch, texturecode_batch, pred_pose_batch3, pred_uv_batch_direct[:, :2, :]

    def pose_regress(self,
                     im_feat_batch,
                     src_pose_batch,
                     tgt_uv_batch,
                     wlh_batch,
                     roi_batch,
                     K_batch):
        src_uv_batch = \
            view_points_batch(corners_of_box_batch(src_pose_batch.detach(), wlh_batch), K_batch, normalize=True)
        # for iter in range(0, iters):

        # normalize src_uv_batch to align with img_in_batch frame, now normalized to (-1, 1) x (-1, 1)
        src_uv_norm, dim_batch = normalize_by_roi(src_uv_batch[:, :2, :], roi_batch, need_square=True)

        # regress delta_pose may be better than delta uv which may not consistent which is more dependant on ransac
        bsize = src_uv_batch.shape[0]
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
        pred_T = torch.matmul(torch.linalg.inv(K_batch), pred_T)

        pred_pose_batch = torch.cat([pred_R, pred_T], dim=2)

        # compute the predict uv from pred_pose, box_dim and K (Tensor version with batch)
        pred_uv_batch = view_points_batch(corners_of_box_batch(pred_pose_batch, wlh_batch), K_batch, normalize=True)

        # ATTENTION: no need to normalize with ROI since the difference is always in the original image pixel scale
        # compute the L2 loss, B x N x 2
        loss = torch.sqrt(torch.sum((pred_uv_batch[:, :2, :] - tgt_uv_batch) ** 2, dim=-2))
        return loss, pred_pose_batch


class TrainerUnifiedNuscenes:
    def __init__(self, save_dir, gpus, nusc_dataset, pretrained_model_dir=None, 
                 resume_from_epoch=None, resume_dir=None,
                 jsonfile='srncar.json', batch_size=2, num_workers=0, shuffle=False,
                 im_enc_rate=1.0, aug_box2d=False, aug_wlh=False, finetune_wlh=False, check_iter=1000):
        """
            The training pipeline for unified pose and nerf model
        """
        super().__init__()
        # Read Hyperparameters
        with open(jsonfile, 'r') as f:
            self.hpams = json.load(f)
        # the device to hold data
        self.device = torch.device('cuda:' + str(0))
        self.nusc_dataset = nusc_dataset
        self.dataloader = DataLoader(self.nusc_dataset, batch_size=batch_size, num_workers=num_workers,
                                     shuffle=shuffle, pin_memory=True, drop_last=True)
        self.batch_size = batch_size
        self.niter, self.nepoch = 0, 0
        self.check_iter = check_iter
        self.im_enc_rate = im_enc_rate
        self.aug_box2d = aug_box2d
        self.aug_wlh = aug_wlh
        self.finetune_wlh = self.hpams['net_hyperparams']['pred_wlh'] and finetune_wlh

        self.make_savedir(save_dir)

        # initialize model
        self.make_model()
        self.parallel_model = torch.nn.DataParallel(
            ParallelModel(self.model, self.hpams, im_enc_rate=im_enc_rate,
                          pred_wlh=self.finetune_wlh), device_ids=list(range(gpus)))

        self.mean_shape = None
        self.mean_texture = None
        if pretrained_model_dir is not None:
            self.load_pretrained_model(pretrained_model_dir)

        # initialize shapecode, texturecode
        self.shape_codes = None
        self.texture_codes = None
        self.instoken2idx = {}
        idx = 0
        for ii, instoken in enumerate(self.nusc_dataset.anntokens_per_ins.keys()):
            if instoken not in self.instoken2idx.keys():
                self.instoken2idx[instoken] = idx
                idx += 1
        self.optimized_idx = torch.zeros(len(self.instoken2idx.keys()))
        self.make_codes()

        # Load from epoch requires initialization before
        if resume_from_epoch is not None:
            self.save_dir = resume_dir
            self.resume_from_epoch(self.save_dir, resume_from_epoch)
        print('we are going to save at ', self.save_dir)

    def train(self, epochs):
        self.t1 = time.time()
        while self.nepoch < epochs:
            print(f'epoch: {self.nepoch}')
            self.set_optimizers()
            self.opts.zero_grad()

            self.training_epoch()

            self.save_models(epoch=self.nepoch)
            self.nepoch += 1

    def training_epoch(self):
        """
            Apply new rendering scheme, rgb loss is computed to size-normalized images
        """

        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f'    Training niter: {self.niter}, epoch: {self.nepoch}, batch: {batch_idx}/{len(self.dataloader)}')

            shapecode_batch = []
            texturecode_batch = []
            for obj_idx, instoken in enumerate(batch_data['instoken']):
                code_idx = self.instoken2idx[instoken]
                code_idx = torch.as_tensor(code_idx)
                shapecode = self.shape_codes(code_idx).unsqueeze(0)
                texturecode = self.texture_codes(code_idx).unsqueeze(0)
                self.optimized_idx[code_idx.item()] = 1
                shapecode_batch.append(shapecode)
                texturecode_batch.append(texturecode)
            shapecode_batch = torch.cat(shapecode_batch).to(self.device)
            texturecode_batch = torch.cat(texturecode_batch).to(self.device)

            img_in_batch = batch_data['img_in'].to(self.device)
            xyz_batch = batch_data['xyz'].to(self.device)
            viewdir_batch = batch_data['viewdir'].to(self.device)
            z_vals_batch = batch_data['z_vals'].to(self.device)
            rgb_tgt_batch = batch_data['rgb_tgt'].to(self.device)
            occ_pixels_batch = batch_data['occ_pixels'].to(self.device)
            roi_batch = batch_data['rois'].to(self.device)

            src_pose_batch = batch_data['obj_poses_w_err'].to(self.device)
            tgt_pose_batch = batch_data['obj_poses'].to(self.device)
            wlh_batch = batch_data['wlh'].to(self.device)
            K_batch = batch_data['cam_intrinsics'].to(self.device)
            tgt_uv_batch = \
                view_points_batch(corners_of_box_batch(tgt_pose_batch.detach(), wlh_batch), K_batch, normalize=True)

            if self.aug_wlh:
                # augment wlh in a volumn preserve way
                wlh_rand_fac = torch.rand(wlh_batch.shape[0], wlh_batch.shape[1]).to(self.device) * 0.2 + 0.9
                wlh_rand_fac[:, 2] = torch.ones(wlh_batch.shape[0]).to(self.device) / wlh_rand_fac[:, 0] / wlh_rand_fac[
                                                                                                           :, 1]
                wlh_batch_aug = wlh_batch * wlh_rand_fac
                tgt_uv_batch_aug = \
                    view_points_batch(corners_of_box_batch(tgt_pose_batch.detach(), wlh_batch_aug), K_batch,
                                      normalize=True)
            else:
                wlh_batch_aug = wlh_batch
                tgt_uv_batch_aug = tgt_uv_batch

            # compute losses from parallel model
            losses_all, loss_total, shapecode_out, texturecode_out, pred_pose_batch, pred_uv_batch_direct = \
                self.parallel_model(img_in_batch,
                                    shapecode_batch,
                                    texturecode_batch,
                                    xyz_batch,
                                    viewdir_batch,
                                    z_vals_batch,
                                    rgb_tgt_batch,
                                    occ_pixels_batch,
                                    src_pose_batch,
                                    tgt_uv_batch[:, :2, :],
                                    wlh_batch,
                                    roi_batch,
                                    K_batch,
                                    wlh_batch_aug,
                                    tgt_uv_batch_aug[:, :2, :]
                                    )
            # compute gradient from the mean loss over multiple gpus
            loss_total.mean().backward()

            # # clip the gradients
            # nn.utils.clip_grad_value_(self.model.parameters(), 1)

            # Recorded codes will be automatically updated from grad
            self.opts.step()
            self.log_losses(losses_all, time.time() - self.t1)

            # reset all the losses
            self.opts.zero_grad()
            self.t1 = time.time()

            # produce visual samples
            if self.niter % self.check_iter == 0:
                num2vis = np.minimum(4, batch_data['imgs'].shape[0])
                src_uv_batch = \
                    view_points_batch(corners_of_box_batch(src_pose_batch.detach(), wlh_batch_aug), K_batch,
                                      normalize=True)
                pred_pose_end = pred_pose_batch[:num2vis]
                pred_uv_end = view_points_batch(corners_of_box_batch(pred_pose_end.detach(), wlh_batch_aug[:num2vis]),
                                                K_batch[:num2vis], normalize=True)
                for ii in range(0, num2vis):
                    tgt_img = batch_data['imgs'][ii]
                    mask_occ = batch_data['masks_occ'][ii]
                    # roi = batch_data['rois'][ii]
                    roi = roi_batch[ii].cpu()
                    K = batch_data['cam_intrinsics'][ii]
                    tgt_pose = batch_data['cam_poses'][ii]
                    anntoken = batch_data['anntoken'][ii]
                    obj_sz = self.nusc_dataset.nusc.get('sample_annotation', anntoken)['size']

                    tgt_img = tgt_img[roi[1]: roi[3], roi[0]: roi[2]]
                    mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)

                    # Just use the cropped region instead to save computation on the visualization
                    with torch.no_grad():
                        generated_img = render_full_img(self.model, self.device, tgt_pose, obj_sz, K, roi,
                                                        self.hpams['n_samples'],
                                                        shapecode_out[ii].unsqueeze(0),
                                                        texturecode_out[ii].unsqueeze(0),
                                                        self.hpams['shapenet_obj_cood'])

                    generated_img = draw_boxes_train(generated_img.cpu().numpy(),
                                                     src_uv_batch[ii, :2, :].cpu().numpy(),
                                                     pred_uv_end[ii, :2, :].cpu().numpy(),
                                                     pred_uv_batch_direct[ii, :2, :].detach().cpu().numpy(),
                                                     roi,
                                                     out_uv=True)
                    generated_img = torch.from_numpy(generated_img)
                    # self.log_img(generated_img, tgt_img, mask_occ, anntoken)
                    self.log_img(generated_img, tgt_img, mask_occ, ii)

            # iterations are only counted after optimized an qualified batch
            self.niter += 1

    def log_losses(self, losses_all, time_spent):
        for key, value in losses_all.items():
            self.writer.add_scalar(key, value.mean().detach().item(), self.niter)
        self.writer.add_scalar('time/train', time_spent, self.niter)

    # def log_losses(self, losses_all, time_spent):
    #     for key, value in losses_all.items():
    #         # self.writer.add_scalar(key, value.mean().detach().item(), self.niter)
    #         wandb.log({
    #             f'Train/{key}': value.mean().detach().item(),
    #         }, step=self.niter)
    #     wandb.log({'Train/time': time_spent}, step=self.niter)

    def log_img(self, generated_img, gtimg, mask_occ, ann_token):
        H, W = generated_img.shape[:-1]
        ret = torch.zeros(H, 2 * W, 3)
        ret[:, :W, :] = generated_img
        ret[:, W:, :] = gtimg * 0.7 + mask_occ * 0.3
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        self.writer.add_image('train_' + str(ann_token),
                              torch.from_numpy(ret).permute(2, 0, 1),
                              global_step=self.niter)
        # wandb.log({f'Train/Visual_{ann_token}': wandb.Image(ret)}, step=self.niter)

    def set_optimizers(self):
        # Always include codes as parameter in case of any updates
        lr1, lr2 = self.get_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': lr1},
            {'params': self.shape_codes.parameters(), 'lr': lr2},
            {'params': self.texture_codes.parameters(), 'lr': lr2}
        ])

    def get_learning_rate(self):
        model_lr, latent_lr = self.hpams['lr_schedule'][0], self.hpams['lr_schedule'][1]
        num_model = self.niter // model_lr['interval']
        num_latent = self.niter // latent_lr['interval']
        lr1 = model_lr['lr'] * 2 ** (-num_model)
        lr2 = latent_lr['lr'] * 2 ** (-num_latent)
        return lr1, lr2

    def make_model(self):
        if self.hpams['arch'] == 'supnerf':
            self.model = SUPNeRF(**self.hpams['net_hyperparams']).to(self.device)
        else:
            print('ERROR: No valid network architecture is declared in config file!')

    def make_codes(self):
        d = len(self.instoken2idx.keys())
        embdim = self.hpams['net_hyperparams']['latent_dim']
        self.shape_codes = nn.Embedding(d, embdim)
        self.texture_codes = nn.Embedding(d, embdim)
        if self.mean_shape is None:
            self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim / 2))
            self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim / 2))
        else:
            self.shape_codes.weight = nn.Parameter(self.mean_shape.repeat(d, 1))
            self.texture_codes.weight = nn.Parameter(self.mean_texture.repeat(d, 1))

    def load_pretrained_model(self, saved_model_file):
        saved_data = torch.load(saved_model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)

        # mean shape should only consider those optimized codes when some of those are not touched
        if 'optimized_idx' in saved_data.keys():
            optimized_idx = saved_data['optimized_idx'].numpy()
            self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'][optimized_idx > 0], dim=0).reshape(1,
                                                                                                                      -1)
            self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'][optimized_idx > 0],
                                           dim=0).reshape(1, -1)
        else:
            self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1, -1)
            self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1, -1)

    def make_savedir(self, save_dir):
        self.save_dir = os.path.join('checkpoints', self.hpams['arch'], 'train_nuscenes_' + save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir, 'runs'))
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'runs'))
        hpampath = os.path.join(self.save_dir, 'hpam.json')
        # update the model folder
        self.hpams['model_dir'] = self.save_dir
        with open(hpampath, 'w') as f:
            json.dump(self.hpams, f, indent=2)

    def save_models(self, iter=None, epoch=None):
        save_dict = {'model_params': self.model.state_dict(),
                     'shape_code_params': self.shape_codes.state_dict(),
                     'texture_code_params': self.texture_codes.state_dict(),
                     'niter': self.niter,
                     'nepoch': self.nepoch,
                     'instoken2idx': self.instoken2idx,
                     'optimized_idx': self.optimized_idx
                     }

        if iter != None:
            torch.save(save_dict, os.path.join(self.save_dir, str(iter) + '.pth'))
        if epoch != None:
            torch.save(save_dict, os.path.join(self.save_dir, f'epoch_{epoch}.pth'))
        torch.save(save_dict, os.path.join(self.save_dir, 'models.pth'))

    def resume_from_epoch(self, saved_dir, epoch):
        print(f'Resume training from saved model at epoch {epoch}.')
        saved_path = os.path.join(saved_dir, f'epoch_{epoch}.pth')
        saved_data = torch.load(saved_path, map_location=torch.device('cpu'))
        # if self.finetune_wlh:
        #     saved_data['model_params']['img_encoder.fc_wlh.weight'] = self.model.img_encoder.fc_wlh.weight
        #     saved_data['model_params']['img_encoder.fc_wlh.bias'] = self.model.img_encoder.fc_wlh.bias
        self.model.load_state_dict(saved_data['model_params'], strict=False)  # only load those both existing keys

        # Check for missing keys manually
        missing_keys = set(self.model.state_dict().keys()) - set(saved_data['model_params'].keys())
        if missing_keys:
            print("The following keys will be finetuned from scratch:", missing_keys)

        self.model = self.model.to(self.device)
        self.niter = saved_data['niter'] + 1
        self.nepoch = saved_data['nepoch'] + 1

        self.shape_codes.load_state_dict(saved_data['shape_code_params'])
        self.texture_codes.load_state_dict(saved_data['texture_code_params'])
        self.instoken2idx = saved_data['instoken2idx']
        self.optimized_idx = saved_data['optimized_idx']
