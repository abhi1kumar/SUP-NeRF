"""
    Adapted for clean training and testing pipeline
    Most for comparing and debugging
"""

import random
import numpy as np
import cv2
import torch
from torchvision.transforms import Resize

from utils import ray_box_intersection, ray_box_intersection_tensor, get_rays, get_rays_specified


class NeRFRenderer(torch.nn.Module):
    def __init__(
        self,
        n_samples=64,
        noise_std=0.0,
        white_bkgd=True,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.white_bkgd = white_bkgd
      
    def sample_from_ray(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_samples
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_samples, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        return near * (1 - z_steps) + far * z_steps  # (B, Kf)

    def volume_render(self, sigmas, rgbs, z_vals):
        """
            rgbs: (n_rays, n_samples, 3)
            sigmas: (n_rays, n_samples)
            z_vals: (n_rays, n_samples)
        """

        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        deltas = torch.cat([deltas, torch.ones_like(deltas[:, :1]) * 1e10], -1)
        alphas = 1 - torch.exp(-torch.relu(sigmas) * deltas)
        trans = 1 - alphas + 1e-10
        transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
        accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
        weights = alphas * accum_trans
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
        depth_final = torch.sum(weights * z_vals, -1)

        if self.white_bkgd:
            # White background
            pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)

        return rgb_final, depth_final, accum_trans[:, -1]

    def volume_render_batch(self, sigmas, rgbs, z_vals):
        """
            consider the first dimension for batch
            rgbs: (B, n_rays, n_samples, 3)
            sigmas: (B, n_rays, n_samples)
            z_vals: (B, n_rays, n_samples)
        """
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas = torch.cat([deltas, torch.ones_like(deltas[..., :1]) * 1e10], -1)
        alphas = 1 - torch.exp(-torch.relu(sigmas).squeeze(-1) * deltas)
        trans = 1 - alphas + 1e-10
        transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
        accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
        weights = alphas * accum_trans
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
        depth_final = torch.sum(weights.unsqueeze(-1) * z_vals.unsqueeze(-1), -2)

        if self.white_bkgd:
            # White background
            pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)

        return rgb_final, depth_final, accum_trans[..., -1]

    def prepare_sampled_rays(self, rays_o, viewdir, obj_sz):
        obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
        obj_w, obj_l, obj_h = obj_sz
        device = rays_o.get_device()
        if device < 0:
            device = 'cpu'
        aabb_min = np.asarray([-obj_l / obj_diag, -obj_w / obj_diag, -obj_h / obj_diag]).reshape((1, 3)).repeat(
            rays_o.shape[0], axis=0).astype(np.float32)
        aabb_max = np.asarray([obj_l / obj_diag, obj_w / obj_diag, obj_h / obj_diag]).reshape((1, 3)).repeat(
            rays_o.shape[0], axis=0).astype(np.float32)

        z_in, z_out, intersect = ray_box_intersection_tensor(rays_o / (obj_diag / 2), viewdir,
                                                             aabb_min=torch.from_numpy(aabb_min).to(device),
                                                             aabb_max=torch.from_numpy(aabb_max).to(device))
        bounds = torch.full_like(rays_o[:, :2], -1)
        bounds[intersect, 0] = z_in
        bounds[intersect, 1] = z_out
        rays = torch.concat(
            [rays_o / (obj_diag / 2), viewdir, bounds], -1)
        z_coarse = self.sample_from_ray(rays)
        xyz = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
        viewdir = viewdir.unsqueeze(-2).repeat(1, self.n_samples, 1)
        # compute z_vals (distance to the camera canter)
        z_vals = torch.norm((xyz - rays[:, None, :3]) * (obj_diag / 2), p=2, dim=-1)
        return xyz, viewdir, z_vals, intersect

    def render_rays(self, model, device, img, mask_occ, cam_pose, obj_sz, K, roi, shapecode, texturecode, kitti2nusc=False, im_sz=64, n_rays=None):
        """
            Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
            Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose

            img : cropped img wrt. roi. already masked before input
            mask_occ: cropped mask_occ wrt. roi
            roi: need to be square (Maybe not necessary) and within img range
        """

        rays_o, viewdir = get_rays(K, cam_pose, roi, uv_steps=[im_sz, im_sz])
        # reshape img and mask_occ to im_sz
        img = img.unsqueeze(0).permute((0, 3, 1, 2))
        img = Resize((im_sz, im_sz))(img)
        img = img.permute((0, 2, 3, 1))
        mask_occ = mask_occ.unsqueeze(0).permute((0, 3, 1, 2))
        mask_occ = Resize((im_sz, im_sz))(mask_occ).type(torch.int32).type(torch.float32)
        mask_occ = mask_occ.permute((0, 2, 3, 1))

        rgb_tgt = img.reshape(-1, 3).to(device)
        occ_pixels = mask_occ.reshape(-1, 1).to(device)

        if n_rays is not None:
            # For different sized roi, extract a random subset of pixels with fixed batch size
            n_rays = np.minimum(rays_o.shape[0], n_rays)
            random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
            rays_o = rays_o[random_ray_ids]
            viewdir = viewdir[random_ray_ids]
            rgb_tgt = rgb_tgt[random_ray_ids]
            occ_pixels = occ_pixels[random_ray_ids]

        # prepare sampled rays in object coordinates
        xyz, viewdir, z_vals, intersect = self.prepare_sampled_rays(
            rays_o.to(device), viewdir.to(device), obj_sz)

        # Kitti to Nuscenes
        if kitti2nusc:
            R_x = np.array([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., -1., 0.]]).astype(np.float32)
            R_x = torch.from_numpy(R_x).view(1, 1, 3, 3).to(device)
            xyz = R_x @ xyz.unsqueeze(-1)
            viewdir = R_x @ viewdir.unsqueeze(-1)
            xyz = xyz.squeeze(-1)
            viewdir = viewdir.squeeze(-1)

        sigmas, rgbs = model(xyz.to(device),
                             viewdir.to(device),
                             shapecode, texturecode)
        rgb_rays, depth_rays, acc_trans_rays = self.volume_render(sigmas.squeeze(), rgbs, z_vals.to(device))
        return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels

    def render_rays_specified(self, model, device, img, mask_occ, cam_pose, obj_sz, K, roi, x_vec, y_vec,
                              shapecode, texturecode, kitti2nusc=False):
        """
            Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
            Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose
        """

        rays_o, viewdir = get_rays_specified(K, cam_pose, x_vec + roi[0].numpy(), y_vec + roi[1].numpy())

        # extract samples
        rgb_tgt = img[y_vec, x_vec, :].to(device)
        occ_pixels = mask_occ[y_vec, x_vec, :].to(device)

        # prepare sampled rays in object coordinates
        xyz, viewdir, z_vals, intersect = self.prepare_sampled_rays(
            rays_o.to(device), viewdir.to(device), obj_sz)

        # Kitti to Nuscenes
        if kitti2nusc:
            R_x = np.array([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., -1., 0.]]).astype(np.float32)
            R_x = torch.from_numpy(R_x).view(1, 1, 3, 3).to(device)
            xyz = R_x @ xyz.unsqueeze(-1)
            viewdir = R_x @ viewdir.unsqueeze(-1)
            xyz = xyz.squeeze(-1)
            viewdir = viewdir.squeeze(-1)

        sigmas, rgbs = model(xyz.to(device),
                             viewdir.to(device),
                             shapecode, texturecode)
        rgb_rays, depth_rays, acc_trans_rays = self.volume_render(sigmas.squeeze(), rgbs, z_vals.to(device))
        return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels

    def prepare_pixel_samples(self, img, mask_occ, cam_pose, obj_sz, K, roi, n_rays, im_sz=None):
        """
            Prepare pixel-sampled data from input image, only one img as input
        """

        if im_sz is None:
            rays_o, viewdir = get_rays(K, cam_pose, roi)
        else:
            rays_o, viewdir = get_rays(K, cam_pose, roi, uv_steps=[im_sz, im_sz])
            # reshape img and mask_occ to im_sz
            img = img.unsqueeze(0).permute((0, 3, 1, 2))
            img = Resize((im_sz, im_sz))(img)
            img = img.permute((0, 2, 3, 1))
            mask_occ = mask_occ.unsqueeze(0).permute((0, 3, 1, 2))
            mask_occ = Resize((im_sz, im_sz))(mask_occ).type(torch.int32).type(torch.float32)
            mask_occ = mask_occ.permute((0, 2, 3, 1))

        # For different sized roi, extract a random subset of pixels with fixed batch size
        n_rays = np.minimum(rays_o.shape[0], n_rays)
        random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
        rays_o = rays_o[random_ray_ids]
        viewdir = viewdir[random_ray_ids]

        # extract samples
        rgb_tgt = img.reshape(-1, 3)[random_ray_ids]
        occ_pixels = mask_occ.reshape(-1, 1)[random_ray_ids]
        mask_rgb = torch.clone(mask_occ)
        mask_rgb[mask_rgb < 0] = 0

        # prepare sampled rays in object coordinates
        xyz, viewdir, z_vals, intersect = self.prepare_sampled_rays(
            rays_o, viewdir, obj_sz)

        return xyz, viewdir, z_vals, rgb_tgt, occ_pixels

    def render_full_img(self, model, device, cam_pose, obj_sz, K, roi, shapecode, texturecode,
                        out_depth=False, debug_occ=False, kitti2nusc=False):
        """
            Assume only one input image, render rgb and depth values of the all the image pixels within the roi area.
            Only the rendered image is returned for visualization purpose.
        """

        rays_o, viewdir = get_rays(K, cam_pose, roi)

        # prepare sampled rays in object coordinates
        xyz, viewdir, z_vals, intersect = self.prepare_sampled_rays(
            rays_o.to(device), viewdir.to(device), obj_sz)

        # Kitti to Nuscenes
        if kitti2nusc:
            R_x = np.array([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., -1., 0.]]).astype(np.float32)
            R_x = torch.from_numpy(R_x).view(1, 1, 3, 3)
            xyz = R_x @ xyz.unsqueeze(-1)
            viewdir = R_x @ viewdir.unsqueeze(-1)
            xyz = xyz.squeeze(-1)
            viewdir = viewdir.squeeze(-1)

        generated_img = []
        generated_depth = []
        generated_acc_trans = []
        sample_step = np.maximum(roi[2] - roi[0], roi[3] - roi[1])
        for i in range(0, xyz.shape[0], sample_step):
            sigmas, rgbs = model(xyz[i:i + sample_step].to(device),
                                 viewdir[i:i + sample_step].to(device),
                                 shapecode, texturecode)

            rgb_rays, depth_rays, acc_trans_rays = self.volume_render(sigmas.squeeze(),
                                                                      rgbs,
                                                                      z_vals[i:i + sample_step].to(device))
            generated_img.append(rgb_rays)
            if out_depth:
                generated_depth.append(depth_rays)
            if debug_occ:
                generated_acc_trans.append(acc_trans_rays)

        generated_img = torch.cat(generated_img).reshape(roi[3] - roi[1], roi[2] - roi[0], 3)

        if debug_occ:
            generated_acc_trans = torch.cat(generated_acc_trans).reshape(roi[3] - roi[1], roi[2] - roi[0])
            cv2.imshow('est_occ',
                       ((torch.ones_like(generated_acc_trans) - generated_acc_trans).cpu().numpy() * 255).astype(
                           np.uint8))
            # cv2.imshow('mask_occ', ((gt_masks_occ[0].cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))
            cv2.waitKey()

        if out_depth:
            generated_depth = torch.cat(generated_depth).reshape(roi[3] - roi[1], roi[2] - roi[0])
            return generated_img, generated_depth

        return generated_img

    def render_virtual_imgs(self, model, device, obj_sz, K, shapecode, texturecode, radius=40.,
                            tilt=np.pi / 6, pan_num=8, img_sz=128, kitti2nusc=False):
        """
            Given NeRF model and conditioned shapecode and texturecode, render a set of virtual images from different views
        """
        virtual_imgs = []
        x_min = K[0, 2] - img_sz / 2
        x_max = K[0, 2] + img_sz / 2
        y_min = K[1, 2] - img_sz / 2
        y_max = K[1, 2] + img_sz / 2
        roi = np.asarray([x_min, y_min, x_max, y_max]).astype(np.int64)
        # sample camera with fixed radius, tilt, and pan angles spanning 2 pi
        cam_init = np.asarray([[0, 0, 1, -radius],
                               [-1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]]).astype(np.float32)
        cam_tilt = np.asarray([[np.cos(tilt), 0, np.sin(tilt), 0],
                               [0, 1, 0, 0],
                               [-np.sin(tilt), 0, np.cos(tilt), 0],
                               [0, 0, 0, 1]]).astype(np.float32) @ cam_init

        pan_angles = np.linspace(0, 2 * np.pi, pan_num, endpoint=False)
        for pan in pan_angles:
            cam_pose = np.asarray([[np.cos(pan), -np.sin(pan), 0, 0],
                                   [np.sin(pan), np.cos(pan), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]]).astype(np.float32) @ cam_tilt
            cam_pose = torch.from_numpy(cam_pose[:3, :])
            generated_img = self.render_full_img(model, device, cam_pose, obj_sz, K, roi,
                                                 shapecode, texturecode,
                                                 kitti2nusc=kitti2nusc)
            # draw the object coordinate basis
            R_w2c = cam_pose[:3, :3].transpose(-1, -2)
            T_w2c = -torch.matmul(R_w2c, cam_pose[:3, 3:])
            P_w2c = torch.cat((R_w2c, T_w2c), dim=1).numpy()
            x_arrow_2d = K @ P_w2c @ torch.asarray([.5, 0., 0., 1.]).reshape([-1, 1])
            x_arrow_2d = (x_arrow_2d[:2] / x_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            y_arrow_2d = K @ P_w2c @ torch.asarray([0., .5, 0., 1.]).reshape([-1, 1])
            y_arrow_2d = (y_arrow_2d[:2] / y_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            z_arrow_2d = K @ P_w2c @ torch.asarray([0., 0., .5, 1.]).reshape([-1, 1])
            z_arrow_2d = (z_arrow_2d[:2] / z_arrow_2d[2]).squeeze().numpy() - K[:2, 2].numpy()
            generated_img = generated_img.cpu().numpy()
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz / 2 + x_arrow_2d[0]), int(img_sz / 2 + x_arrow_2d[1])),
                                            (1, 0, 0))
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz / 2 + y_arrow_2d[0]), int(img_sz / 2 + y_arrow_2d[1])),
                                            (0, 1, 0))
            generated_img = cv2.arrowedLine(generated_img,
                                            (int(img_sz / 2), int(img_sz / 2)),
                                            (int(img_sz / 2 + z_arrow_2d[0]), int(img_sz / 2 + z_arrow_2d[1])),
                                            (0, 0, 1))
            virtual_imgs.append(torch.from_numpy(generated_img))

        return virtual_imgs


def volume_rendering3(sigmas, rgbs, z_vals, white_bkgd=False):
    """
        return accumulated transparency in addition
        each ray has its own z_vals, the first dim of z_vals equals to sigmas and rgbs
        TODO: the returned depth should be in z-buffer for evaluation
        TODO: pad the last deltas with 0 leads to black BG, might be better for removing floating --> last weight = 0

    """
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:, :1]) * 1e10], -1)
    # deltas = torch.cat([deltas, torch.zeros_like(deltas[:, :1])], -1)
    alphas = 1 - torch.exp(-torch.relu(sigmas).squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)

    if white_bkgd:
        # White background
        pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
        rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)

    return rgb_final, depth_final, accum_trans[:, -1]


def render_rays_v3(model, device, img, mask_occ, cam_pose, obj_wlh, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, sym_aug, kitti2nusc=False, im_sz=64, n_rays=None, adjust_scale=1.0):
    """
        Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
        Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose

        img : cropped img wrt. roi. already masked before input
        mask_occ: cropped mask_occ wrt. roi
        roi: need to be square (Maybe not necessary) and within img range

        v3: This version apply ray-AABB intersection to performance more accurate ray sampling
    """
    # obj_diag = obj_diag * 1.1
    renderer = NeRFRenderer()

    rays_o, viewdir = get_rays(K, cam_pose, roi, uv_steps=[im_sz, im_sz])
    # reshape img and mask_occ to im_sz
    img = img.unsqueeze(0).permute((0, 3, 1, 2))
    img = Resize((im_sz, im_sz))(img)
    img = img.permute((0, 2, 3, 1))
    mask_occ = mask_occ.unsqueeze(0).permute((0, 3, 1, 2))
    mask_occ = Resize((im_sz, im_sz))(mask_occ).type(torch.int32).type(torch.float32)
    mask_occ = mask_occ.permute((0, 2, 3, 1))

    rgb_tgt = img.reshape(-1, 3).to(device)
    occ_pixels = mask_occ.reshape(-1, 1).to(device)

    if n_rays is not None:
        # For different sized roi, extract a random subset of pixels with fixed batch size
        n_rays = np.minimum(rays_o.shape[0], n_rays)
        random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
        rays_o = rays_o[random_ray_ids]
        viewdir = viewdir[random_ray_ids]
        rgb_tgt = rgb_tgt[random_ray_ids]
        occ_pixels = occ_pixels[random_ray_ids]

    obj_diag = np.linalg.norm(obj_wlh).astype(np.float32)
    obj_w, obj_l, obj_h = obj_wlh
    aabb_min = np.asarray([-obj_l / obj_diag, -obj_w / obj_diag, -obj_h / obj_diag]).reshape((1, 3)).repeat(
        rays_o.shape[0], axis=0)
    aabb_max = np.asarray([obj_l / obj_diag, obj_w / obj_diag, obj_h / obj_diag]).reshape((1, 3)).repeat(
        rays_o.shape[0], axis=0)
    # aabb_min = None
    # aabb_max = None
    z_in, z_out, intersect = ray_box_intersection(rays_o.cpu().detach().numpy() / (obj_diag / 2),
                                                  viewdir.cpu().detach().numpy(),
                                                  aabb_min=aabb_min,
                                                  aabb_max=aabb_max
                                                  )
    bounds = np.ones((*rays_o.shape[:-1], 2)) * -1
    bounds[intersect, 0] = z_in
    bounds[intersect, 1] = z_out
    rays = torch.concat([rays_o.to(device) / (obj_diag / 2), viewdir.to(device), torch.FloatTensor(bounds).to(device)], -1)
    z_coarse = renderer.sample_from_ray(rays)
    xyz = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
    viewdir = viewdir.unsqueeze(-2).repeat(1, n_samples, 1)
    # compute z_vals (distance to the camera canter)
    z_vals = torch.norm((xyz - rays[:, None, :3]) * (obj_diag / 2), p=2, dim=-1)

    # adjust scale to match trained models
    xyz *= adjust_scale
    # Apply symmetric augmentation
    if sym_aug and random.uniform(0, 1) > 0.5:
        xyz[:, :, 1] *= (-1)
        viewdir[:, :, 1] *= (-1)

    # Kitti to Nuscenes
    if kitti2nusc:
        R_x = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., -1., 0.]]).astype(np.float32)
        R_x = torch.from_numpy(R_x).view(1, 1, 3, 3).to(device)
        xyz = R_x @ xyz.unsqueeze(-1)
        viewdir = R_x @ viewdir.unsqueeze(-1)
        xyz = xyz.squeeze(-1)
        viewdir = viewdir.squeeze(-1)

    # Nuscene to ShapeNet: frame rotate -90 degree around Z, coord rotate 90 degree around Z
    if shapenet_obj_cood:
        xyz = xyz[:, :, [1, 0, 2]]
        xyz[:, :, 0] *= (-1)
        viewdir = viewdir[:, :, [1, 0, 2]]
        viewdir[:, :, 0] *= (-1)

    sigmas, rgbs = model(xyz.to(device),
                         viewdir.to(device),
                         shapecode, texturecode)
    rgb_rays, depth_rays, acc_trans_rays = volume_rendering3(sigmas, rgbs, z_vals.to(device))
    # rgb_rays, depth_rays, acc_trans_rays = renderer.volume_render(rgbs, sigmas.squeeze(), z_vals.to(device))

    # Should occ_pixels updated with intersect? --no, the occ mask should be more accurate.
    # occ_pixels[~intersect] = 0
    return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels