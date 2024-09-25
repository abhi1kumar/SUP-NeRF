import os.path
import random
import numpy as np
import cv2
import matplotlib
import torch
import argparse
import torchvision
from torchvision.transforms import Resize
from scipy.spatial.transform import Rotation as R


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def preprocess_img_keepratio(img, max_img_sz=320):
    """
        keep dim, and resize the max dim to max_img_sz if over
    """
    img = img.unsqueeze(0).permute((0, 3, 1, 2))
    _, _, im_h, im_w = img.shape
    if np.maximum(im_h, im_w) > max_img_sz:
        ratio = max_img_sz / np.maximum(im_h, im_w)
        new_h = im_h * ratio
        new_w = im_w * ratio
        img = Resize((int(new_h), int(new_w)))(img)
    return img


def preprocess_img_square(img, new_size=128, pad_white=True):
    """
        The largest dim resize to new_size, pad the other to make square
        Make the padding area white
    """
    img = img.unsqueeze(0).permute((0, 3, 1, 2))
    _, _, im_h, im_w = img.shape
    ratio = new_size / np.maximum(im_h, im_w)
    new_h = int(im_h * ratio)
    new_w = int(im_w * ratio)
    img = Resize((new_h, new_w))(img)
    if pad_white:
        new_img = torch.ones((1, 3, new_size, new_size), dtype=torch.float32)
    else:
        new_img = torch.zeros((1, 3, new_size, new_size), dtype=torch.float32)
    y_start = int(new_size/2 - new_h/2)
    x_start = int(new_size/2 - new_w/2)

    new_img[:, :, y_start: y_start + new_h, x_start: x_start + new_w] = img
    return new_img


def preprocess_occ_square(occ_mask, new_size=128, pad_value=-1):
    """
        The largest dim resize to new_size, pad the other to make square
        pad with assign value
    """
    im_h, im_w = occ_mask.shape
    occ_mask = occ_mask.view((1, 1, im_h, im_w))
    ratio = new_size / np.maximum(im_h, im_w)
    new_h = int(im_h * ratio)
    new_w = int(im_w * ratio)
    occ_mask = Resize((new_h, new_w))(occ_mask)
    new_img = torch.ones((1, 1, new_size, new_size), dtype=torch.float32) * pad_value
    y_start = int(new_size/2 - new_h/2)
    x_start = int(new_size/2 - new_w/2)

    new_img[:, :, y_start: y_start + new_h, x_start: x_start + new_w] = occ_mask
    return torch.floor(new_img)


def get_rays_srn(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    # TODO: such dir seems to be based on wield camera pose c2w and camera frame definition
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


def get_rays(K, c2w, roi, uv_steps=None):
    """
        K: intrinsic matrix
        c2w: camera pose in object (world) coordinate frame
        roi: [min_x, min_y, max_x, max_y]

        ATTENTION:
        the number of output rays depends on roi inputs
        nuscenes uses a different camera coordinate frame compared to shapenet srn
    """
    dx = K[0, 2]
    dy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    if uv_steps is not None:
        i, j = torch.meshgrid(torch.linspace(roi[0], roi[2] - 1, uv_steps[0]),
                              torch.linspace(roi[1], roi[3] - 1, uv_steps[1]))
    else:
        i, j = torch.meshgrid(torch.linspace(roi[0], roi[2]-1, roi[2]-roi[0]),
                              torch.linspace(roi[1], roi[3]-1, roi[3]-roi[1]))
    i = i.t()
    j = j.t()
    # some signs are opposite to get_rays for shapenet srn dataset
    dirs = torch.stack([(i - dx) / fx, (j - dy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


def get_rays_specified(K, c2w, x_vec, y_vec):
    dx = K[0, 2]
    dy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    i = torch.from_numpy(x_vec).t()
    j = torch.from_numpy(y_vec).t()
    # some signs are opposite to get_rays for shapenet srn dataset
    dirs = torch.stack([(i - dx) / fx, (j - dy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs


def sample_from_rays(ro, vd, near, far, N_samples, z_fixed=False):
    # Given ray centre (camera location), we sample z_vals
    # TODO: this type of sampling might be limited to the camera view facing the object center.
    # TODO: output z_vals are expected to be used distance from ray, this simple way can only generate samples between two circular surfaces, may not cover the object well
    # we do not use ray_o here - just number of rays
    if z_fixed:
        z_vals = torch.linspace(near, far, N_samples).type_as(ro)
    else:
        dist = (far - near) / (2*N_samples)
        z_vals = torch.linspace(near+dist, far-dist, N_samples).type_as(ro)
        z_vals += (torch.rand(N_samples) * (far - near) / (2*N_samples)).type_as(ro)
    xyz = ro.unsqueeze(-2) + vd.unsqueeze(-2) * z_vals.unsqueeze(-1)
    vd = vd.unsqueeze(-2).repeat(1,N_samples,1)
    return xyz, vd, z_vals


def sample_from_rays_v2(rays, n_samples):
    """
    Stratified sampling. Note this is different from original NeRF slightly.
    :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
    :return (B, Kc)
    """
    device = rays.device
    near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

    step = 1.0 / n_samples
    B = rays.shape[0]
    z_steps = torch.linspace(0, 1 - step, n_samples, device=device)  # (Kc)
    z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
    z_steps += torch.rand_like(z_steps) * step
    return near * (1 - z_steps) + far * z_steps  # (B, Kf)


def volume_rendering(sigmas, rgbs, z_vals):
    # TODO: the returned depth should be in z-buffer for evaluation
    deltas = z_vals[1:] - z_vals[:-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:1]) * 1e10])
    alphas = 1 - torch.exp(-sigmas.squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)

    return rgb_final, depth_final


def volume_rendering2(sigmas, rgbs, z_vals):
    """
        return accumulated transparency in addition
        ATTENTION: z_vals in theoray should be the distance to camera center, not z-buffer
        TODO: the returned depth should be in z-buffer for evaluation
    """
    deltas = z_vals[1:] - z_vals[:-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:1]) * 1e10])
    alphas = 1 - torch.exp(-torch.relu(sigmas).squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)
    return rgb_final, depth_final, accum_trans[:, -1]


def volume_rendering_batch(sigmas, rgbs, z_vals):
    """
        consider the first dimension for batch
    """
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:, :1]) * 1e10], -1)
    alphas = 1 - torch.exp(-torch.relu(sigmas).squeeze(-1) * deltas.unsqueeze(1))
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals.unsqueeze(1), -1)
    return rgb_final, depth_final, accum_trans[:, :, -1]


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = np.ones_like(ray_o) * -1.  # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = np.ones_like(ray_o)  # tf.constant([1., 1., 1.])

    inv_d = np.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = np.minimum(t_min, t_max)
    t1 = np.maximum(t_min, t_max)

    t_near = np.maximum(np.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = np.minimum(np.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = t_far > t_near  # np.where(t_far > t_near)[0]

    # Check that boxes are in front of the ray origin
    positive_far = (t_far * intersection_map) > 0
    intersection_map = np.logical_and(intersection_map, positive_far)

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def ray_box_intersection_tensor(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.full_like(ray_o, -1.)   # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.full_like(ray_o, 1.)  # tf.constant([1., 1., 1.])

    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = t_far > t_near  # np.where(t_far > t_near)[0]

    # Check that boxes are in front of the ray origin
    positive_far = (t_far * intersection_map) > 0
    intersection_map = torch.logical_and(intersection_map, positive_far)

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def prepare_pixel_samples(img, mask_occ, cam_pose, obj_diag, K, roi, n_rays, n_samples, shapenet_obj_cood, sym_aug, im_sz=None):
    """
        Prepare pixel-sampled data from input image, only one img as input
    """
    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

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

    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
    xyz /= obj_diag

    # Apply symmetric augmentation
    if sym_aug and random.uniform(0, 1) > 0.5:
        xyz[:, :, 1] *= (-1)
        viewdir[:, :, 1] *= (-1)

    # Nuscene to ShapeNet: frame rotate -90 degree around Z, coord rotate 90 degree around Z
    if shapenet_obj_cood:
        xyz = xyz[:, :, [1, 0, 2]]
        xyz[:, :, 0] *= (-1)
        viewdir = viewdir[:, :, [1, 0, 2]]
        viewdir[:, :, 0] *= (-1)

    return xyz, viewdir, z_vals, rgb_tgt, occ_pixels


def render_rays(model, device, img, mask_occ, cam_pose, obj_diag, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, sym_aug, kitti2nusc=False, n_rays=2500):
    """
        Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
        Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose
    """
    # obj_diag = obj_diag * 1.1

    rays_o, viewdir = get_rays(K, cam_pose, roi)

    # For different sized roi, extract a random subset of pixels with fixed batch size
    n_rays = np.minimum(rays_o.shape[0], n_rays)
    random_ray_ids = np.random.permutation(rays_o.shape[0])[:n_rays]
    rays_o = rays_o[random_ray_ids]
    viewdir = viewdir[random_ray_ids]

    # extract samples
    rgb_tgt = img.reshape(-1, 3)[random_ray_ids].to(device)
    occ_pixels = mask_occ.reshape(-1, 1)[random_ray_ids].to(device)

    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2
    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
    xyz /= obj_diag

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
    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
    return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels


def render_rays_v2(model, device, img, mask_occ, cam_pose, obj_diag, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, sym_aug, kitti2nusc=False, im_sz=64, n_rays=None):
    """
        Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
        Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose

        img : cropped img wrt. roi. already masked before input
        mask_occ: cropped mask_occ wrt. roi
        roi: need to be square (Maybe not necessary) and within img range
    """
    # obj_diag = obj_diag * 1.1

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

    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2

    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
    xyz /= obj_diag  # this might be the actual bug? should be xyz /= (obj_diag / 2)

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
    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
    # Should occ_pixels updated with intersect? --no, the occ mask should be more accurate.
    return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels

def render_rays_specified(model, device, img, mask_occ, cam_pose, obj_diag, K, roi, x_vec, y_vec, n_samples, shapecode, texturecode,
                          shapenet_obj_cood, sym_aug, kitti2nusc=False):
    """
        Assume only one input image, sample pixels from the roi area, and render rgb and depth values of the sampled pixels.
        Return both rendered values and tgt values for the sampled pixels, as well additional output for training purpose
    """
    # obj_diag = obj_diag * 1.1

    rays_o, viewdir = get_rays_specified(K, cam_pose, x_vec + roi[0].numpy(), y_vec + roi[1].numpy())

    # extract samples
    rgb_tgt = img[y_vec, x_vec, :].to(device)
    occ_pixels = mask_occ[y_vec, x_vec, :].to(device)

    # near and far sample range need to be adaptively calculated
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2
    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
    xyz /= obj_diag

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
    rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
    return rgb_rays, depth_rays, acc_trans_rays, rgb_tgt, occ_pixels


def render_full_img(model, device, cam_pose, obj_sz, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, out_depth=False, debug_occ=False, kitti2nusc=False):
    """
        Assume only one input image, render rgb and depth values of the all the image pixels within the roi area.
        Only the rendered image is returned for visualization purpose.
    """
    obj_diag = np.linalg.norm(obj_sz).astype(np.float32)
    # obj_diag = obj_diag * 1.1

    rays_o, viewdir = get_rays(K, cam_pose, roi)

    # how to better define near and far sample range
    near = np.linalg.norm(cam_pose[:, -1].tolist()) - obj_diag / 2
    far = np.linalg.norm(cam_pose[:, -1].tolist()) + obj_diag / 2
    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, near, far, n_samples)
    xyz /= obj_diag

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

    # Nuscene to ShapeNet: rotate 90 degree around Z
    if shapenet_obj_cood:
        xyz = xyz[:, :, [1, 0, 2]]
        xyz[:, :, 0] *= (-1)
        viewdir = viewdir[:, :, [1, 0, 2]]
        viewdir[:, :, 0] *= (-1)

    generated_img = []
    generated_depth = []
    generated_acc_trans = []
    sample_step = np.maximum(roi[2] - roi[0], roi[3] - roi[1])
    for i in range(0, xyz.shape[0], sample_step):
        sigmas, rgbs = model(xyz[i:i + sample_step].to(device),
                                  viewdir[i:i + sample_step].to(device),
                                  shapecode, texturecode)

        rgb_rays, depth_rays, acc_trans_rays = volume_rendering2(sigmas, rgbs, z_vals.to(device))
        generated_img.append(rgb_rays)
        if out_depth:
            generated_depth.append(depth_rays)
        if debug_occ:
            generated_acc_trans.append(acc_trans_rays)

    generated_img = torch.cat(generated_img).reshape(roi[3] - roi[1], roi[2] - roi[0], 3)

    if debug_occ:
        generated_acc_trans = torch.cat(generated_acc_trans).reshape(roi[3]-roi[1], roi[2]-roi[0])
        cv2.imshow('est_occ', ((torch.ones_like(generated_acc_trans) - generated_acc_trans).cpu().numpy() * 255).astype(np.uint8))
        # cv2.imshow('mask_occ', ((gt_masks_occ[0].cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))
        cv2.waitKey()

    if out_depth:
        generated_depth = torch.cat(generated_depth).reshape(roi[3] - roi[1], roi[2] - roi[0])
        return generated_img, generated_depth

    return generated_img


def render_virtual_imgs(model, device, obj_sz, K, n_samples, shapecode, texturecode, shapenet_obj_cood, radius=40., tilt=np.pi/6, pan_num=8, img_sz=128, kitti2nusc=False):
    """
        Given NeRF model and conditioned shapecode and texturecode, render a set of virtual images from different views
    """
    virtual_imgs = []
    x_min = K[0, 2] - img_sz/2
    x_max = K[0, 2] + img_sz/2
    y_min = K[1, 2] - img_sz/2
    y_max = K[1, 2] + img_sz/2
    roi = np.asarray([x_min, y_min, x_max, y_max]).astype(np.int)
    # sample camera with fixed radius, tilt, and pan angles spanning 2 pi
    cam_init = np.asarray([[0,   0,  1, -radius],
                           [-1,  0,  0, 0],
                           [0,  -1,  0, 0],
                           [0,   0,  0, 1]]).astype(np.float32)
    cam_tilt = np.asarray([[np.cos(tilt),   0, np.sin(tilt), 0],
                           [0,              1, 0,             0],
                           [-np.sin(tilt),   0, np.cos(tilt),  0],
                           [0,              0, 0,             1]]).astype(np.float32) @ cam_init

    pan_angles = np.linspace(0, 2*np.pi, pan_num, endpoint=False)
    for pan in pan_angles:
        cam_pose = np.asarray([[np.cos(pan),   -np.sin(pan), 0, 0],
                               [np.sin(pan),   np.cos(pan),  0, 0],
                               [0,              0,           1, 0],
                               [0,              0,           0, 1]]).astype(np.float32) @ cam_tilt
        cam_pose = torch.from_numpy(cam_pose[:3, :])
        generated_img = render_full_img(model, device, cam_pose, obj_sz, K, roi, n_samples, shapecode, texturecode, shapenet_obj_cood, kitti2nusc=kitti2nusc)
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
                                        (int(img_sz/2 + x_arrow_2d[0]), int(img_sz/2 + x_arrow_2d[1])),
                                        (1, 0, 0))
        generated_img = cv2.arrowedLine(generated_img,
                                        (int(img_sz / 2), int(img_sz / 2)),
                                        (int(img_sz/2 + y_arrow_2d[0]), int(img_sz/2 + y_arrow_2d[1])),
                                        (0, 1, 0))
        generated_img = cv2.arrowedLine(generated_img,
                                        (int(img_sz / 2), int(img_sz / 2)),
                                        (int(img_sz/2 + z_arrow_2d[0]), int(img_sz/2 + z_arrow_2d[1])),
                                        (0, 0, 1))
        virtual_imgs.append(torch.from_numpy(generated_img))

    return virtual_imgs


def calc_pose_err(est_poses, tgt_poses):
    est_R = est_poses[:, :3, :3]
    est_T = est_poses[:, :3, 3]
    tgt_R = tgt_poses[:, :3, :3]
    tgt_T = tgt_poses[:, :3, 3]

    err_R = rot_dist(est_R, tgt_R)
    err_T = torch.sqrt(torch.sum((est_T - tgt_T) ** 2, dim=-1))
    return err_R, err_T


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    # #print(img.shape)
    # vmin = np.min(img)
    # vmax = np.max(img)
    # if vmax - vmin < 1e-10:
    #     vmax += 1e-10
    # img = (img - vmin) / (vmax - vmin)
    img[img < 0] = 0
    img[img > 1] = 1
    img *= 255.0
    return img.astype(np.uint8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rot_dist(R1, R2):
    """
    R1: B x 3 x 3
    R2: B x 3 x 3
    return B X 1
    """
    R_diff = torch.matmul(R1, torch.transpose(R2, -1, -2))
    trace = torch.tensor([torch.trace(R_diff_single) for R_diff_single in R_diff])
    trace = torch.maximum(torch.tensor(-1), torch.minimum(torch.tensor(3), trace))
    return torch.acos((trace-1) / 2)


def generate_obj_sz_reg_samples(obj_sz, obj_diag, shapenet_obj_cood=True, tau=0.05, samples_per_plane=100):
    """
        Generate samples around limit planes
    """
    norm_limits = obj_sz / obj_diag
    if shapenet_obj_cood:
        norm_limits = norm_limits[[1, 0, 2]]  # the limit does not care the sign
    x_lim, y_lim, z_lim = norm_limits
    out_samples = {}
    X = np.random.uniform(-x_lim, x_lim, samples_per_plane)
    Y = np.random.uniform(-y_lim, y_lim, samples_per_plane)
    Z = np.random.uniform(-z_lim, z_lim, samples_per_plane)

    out_samples['X_planes_out'] = np.concatenate([np.asarray([np.ones(samples_per_plane) * (-x_lim - tau), Y, Z]).transpose(),
                                                np.asarray([np.ones(samples_per_plane) * (x_lim + tau), Y, Z]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['X_planes_in'] = np.concatenate([np.asarray([np.ones(samples_per_plane) * (-x_lim + tau), Y, Z]).transpose(),
                                                np.asarray([np.ones(samples_per_plane) * (x_lim - tau), Y, Z]).transpose()],
                                               axis=0).astype(np.float32)

    out_samples['Y_planes_out'] = np.concatenate([np.asarray([X, np.ones(samples_per_plane) * (-y_lim - tau), Z]).transpose(),
                                                np.asarray([X, np.ones(samples_per_plane) * (y_lim + tau), Z]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['Y_planes_in'] = np.concatenate([np.asarray([X, np.ones(samples_per_plane) * (-y_lim + tau), Z]).transpose(),
                                                np.asarray([X, np.ones(samples_per_plane) * (y_lim - tau), Z]).transpose()],
                                               axis=0).astype(np.float32)

    out_samples['Z_planes_out'] = np.concatenate([np.asarray([X, Y, np.ones(samples_per_plane) * (-z_lim - tau)]).transpose(),
                                                np.asarray([X, Y, np.ones(samples_per_plane) * (z_lim + tau)]).transpose()],
                                               axis=0).astype(np.float32)
    out_samples['Z_planes_in'] = np.concatenate([np.asarray([X, Y, np.ones(samples_per_plane) * (-z_lim + tau)]).transpose(),
                                                np.asarray([X, Y, np.ones(samples_per_plane) * (z_lim - tau)]).transpose()],
                                               axis=0).astype(np.float32)
    return out_samples


def align_imgs_width(imgs, W, max_view=4):
    """
        imgs: a list of tensors
    """
    out_imgs = []
    if len(imgs) > max_view:
        step = len(imgs) // max_view
    else:
        step = 1

    for id in range(0, len(imgs), step):
        img = imgs[id]
        H_i, W_i = img.shape[:2]
        H_out = int(float(H_i) * W / W_i)
        # out_imgs.append(Image.fromarray(img.detach().cpu().numpy()).resize((W, H_out)))
        img = img.reshape((1, H_i, W_i, -1))
        img = img.permute((0, 3, 1, 2))
        img = torchvision.transforms.Resize((H_out, W))(img)
        img = img.permute((0, 2, 3, 1))
        out_imgs.append(img.squeeze())
        if len(out_imgs) == max_view:
            break
    return out_imgs


def collect_eval_results(result_file, max_iter, axes, color, opt_pose, cross_eval_file=None,
                         print_iters=[0, 3, 5, 10, 20, 50, 99], rot_outlier_ignore=False,
                         sample_keys=None):
    print(f'Processing {result_file}')
    saved_result = torch.load(result_file, map_location=torch.device('cpu'))

    iters = np.array([i for i in range(0, max_iter)])
    psnr_all = []
    depth_err_mean_all = []
    lidar_pts_cnt_all = []

    if sample_keys is not None:
        for key in sample_keys:
            psnr = saved_result['psnr_eval'][key]
            psnr_all.append(np.array(psnr)[:max_iter])
            if key in saved_result['depth_err_mean'].keys():
                depth_err_mean = saved_result['depth_err_mean'][key]
                depth_err_mean_all.append(np.array(depth_err_mean)[:max_iter])
                lidar_pts_cnt = saved_result['lidar_pts_cnt'][key]
                lidar_pts_cnt_all.append(lidar_pts_cnt)
    else:
        for psnr in saved_result['psnr_eval'].values():
            psnr_all.append(np.array(psnr)[:max_iter])
        for depth_err_mean in saved_result['depth_err_mean'].values():
            depth_err_mean_all.append(np.array(depth_err_mean)[:max_iter])
        for lidar_pts_cnt in saved_result['lidar_pts_cnt'].values():
            lidar_pts_cnt_all.append(lidar_pts_cnt)

    psnr_all = np.asarray(psnr_all)
    # deal with inf in psnr
    # TODO: such dealing with psnr might be biased
    psnr_all[np.argwhere(np.isinf(psnr_all))] = 0
    psnr_all[psnr_all < 0] = 0
    psnr_iters = np.mean(psnr_all, axis=0)
    psnr_print = np.round(psnr_iters[print_iters], 2)
    print(f'    psnr: {psnr_print}')

    if len(depth_err_mean_all) > 0:
        depth_err_mean_all = np.asarray(depth_err_mean_all)
        lidar_pts_cnt_all = np.asarray(lidar_pts_cnt_all)

        depth_err_iters = np.sum(depth_err_mean_all * np.expand_dims(lidar_pts_cnt_all, -1), axis=0) / np.sum(
            lidar_pts_cnt_all)

        depth_err_print = np.round(depth_err_iters[print_iters], 2)
        print(f'    depth err: {depth_err_print}')

    lines = []
    # if opt_pose > 0:
    line, = axes[0, 0].plot(iters, psnr_iters, f'{color}-', linewidth=2)
    # axes[0, 0].fill_between(iters, psnr_iters-psnr_iters_std, psnr_iters+psnr_iters_std, alpha=0.2, facecolor=f'{color}')
    axes[0, 0].set_title('PSNR')
    axes[0, 0].set_xlabel('Iters')
    axes[0, 0].set_ylabel('PSNR')
    lines.append(line)

    if len(depth_err_mean_all) > 0:
        line, = axes[0, 1].plot(iters, depth_err_iters, f'{color}-', linewidth=2)
        # axes[0, 1].fill_between(iters, depth_err_iters-depth_err_iters_std, depth_err_iters+depth_err_iters_std, alpha=0.2, facecolor=f'{color}')
        axes[0, 1].set_title('Depth Err')
        axes[0, 1].set_xlabel('Iters')
        axes[0, 1].set_ylabel('Meters')
        lines.append(line)

    R_err_all = []
    T_err_all = []
    if sample_keys is not None:
        for key in sample_keys:
            R_err = saved_result['R_eval'][key]
            R_err_all.append(torch.stack(R_err).numpy()[:max_iter].tolist())
            T_err = saved_result['T_eval'][key]
            T_err_all.append(torch.stack(T_err).numpy()[:max_iter].tolist())
    else:
        for R_err in saved_result['R_eval'].values():
            R_err_all.append(torch.stack(R_err).numpy()[:max_iter].tolist())
        for T_err in saved_result['T_eval'].values():
            T_err_all.append(torch.stack(T_err).numpy()[:max_iter].tolist())

    R_err_all = np.asarray(R_err_all)
    # Deal with NaN in Rot, just no error
    R_err_all[np.argwhere(np.isnan(R_err_all))] = 0
    if rot_outlier_ignore:
        # TODO: the right way is to flip 180 in rot, not direct reduce the error. Other cases may benefit
        R_err_all_r0_copy = np.copy(R_err_all[:, 0])
        rot_flip_ratio_last = len(np.argwhere(R_err_all[:, -1] > np.pi * 0.9)) / R_err_all.shape[0]
        print(f'rot_flip_ratio_last: {rot_flip_ratio_last}')
        R_err_all[R_err_all > np.pi * 0.9] = np.abs(R_err_all[R_err_all > np.pi * 0.9] - np.pi)
        R_err_all[:, 0] = R_err_all_r0_copy
    T_err_all = np.asarray(T_err_all)

    R_err_iters = np.mean(R_err_all, axis=0)
    R_err_iters_std = np.std(R_err_all, axis=0)
    T_err_iters = np.mean(T_err_all, axis=0)

    R_err_iters = R_err_iters / np.pi * 180
    line, = axes[1, 0].plot(iters, R_err_iters, f'{color}-', linewidth=2)
    # axes[1, 0].fill_between(iters, R_err_iters-R_err_iters_std, R_err_iters+R_err_iters_std, alpha=0.2, facecolor=f'{color}')
    axes[1, 0].set_title('Rot Err')
    axes[1, 0].set_xlabel('Iters')
    axes[1, 0].set_ylabel('Degree')
    lines.append(line)
    R_err_print = np.round(R_err_iters[print_iters], 2)
    print(f'    R err: {R_err_print}')

    line, = axes[1, 1].plot(iters, T_err_iters, f'{color}-', linewidth=2)
    # axes[1, 1].fill_between(iters, T_err_iters-T_err_iters_std, T_err_iters+T_err_iters_std, alpha=0.2, facecolor=f'{color}')
    axes[1, 1].set_title('Trans Err')
    axes[1, 1].set_xlabel('Iters')
    axes[1, 1].set_ylabel('Meters')
    lines.append(line)
    T_err_print = np.round(T_err_iters[print_iters], 2)
    print(f'    T err: {T_err_print}')
    # else:
    #     if len(axes.shape) > 1:
    #         axe = axes[0, 0]
    #     else:
    #         axe = axes[0]
    #     line, = axe.plot(iters, psnr_iters, f'{color}-', linewidth=2)
    #     # axes[0].fill_between(iters, psnr_iters-psnr_iters_std, psnr_iters+psnr_iters_std, alpha=0.2, facecolor=f'{color}')
    #     axe.set_title('PSNR')
    #     axe.set_xlabel('Iters')
    #     axe.set_ylabel('PSNR')
    #     lines.append(line)
    #
    #     if len(axes.shape) > 1:
    #         axe = axes[0, 1]
    #     else:
    #         axe = axes[1]
    #     if len(depth_err_mean_all) > 0:
    #         line, = axe.plot(iters, depth_err_iters, f'{color}-', linewidth=2)
    #         # axes[1].fill_between(iters, depth_err_iters-depth_err_iters_std, depth_err_iters+depth_err_iters_std, alpha=0.2, facecolor=f'{color}')
    #         axe.set_title('Depth Error')
    #         axe.set_xlabel('Iters')
    #         axe.set_ylabel('Meters')
    #         lines.append(line)

    """
        Include cross-view evaluation if available
    """

    if cross_eval_file is not None and os.path.exists(cross_eval_file):
        cross_eval_result = torch.load(cross_eval_file, map_location=torch.device('cpu'))
        code_save_iters = cross_eval_result['CODE_SAVE_ITERS_']
        print(f'    code save iters cross-view: {code_save_iters}')
        n_iters = len(code_save_iters)
        n_ins_multiview = 0
        # first sift out those instance with multi-view data
        for ins_id in cross_eval_result['psnr_eval_mat_per_ins'].keys():
            # cnt_lidar_pts_per_ins = cross_eval_result['cnt_lidar_pts_per_ins'][ins_id]
            # num_cams = len(cnt_lidar_pts_per_ins[0])
            num_cams = cross_eval_result['psnr_eval_mat_per_ins'][ins_id][0].shape[0]
            if num_cams < 2:
                continue
            n_ins_multiview += 1

        psnr_cross_eval_all = {}
        depth_cross_eval_all = {}
        # cnt_lidar_cross_eval_all = {}
        for iter_i in range(0, n_iters):
            psnr_cross_eval_all[iter_i] = []
            depth_cross_eval_all[iter_i] = []
            # cnt_lidar_cross_eval_all[iter_i] = []

        for ins_id in cross_eval_result['psnr_eval_mat_per_ins'].keys():
            num_cams = cross_eval_result['psnr_eval_mat_per_ins'][ins_id][0].shape[0]
            if num_cams < 2:
                continue
            for iter_i in range(0, n_iters):
                # cnt_lidar_pts_per_ins_per_iter = cross_eval_result['cnt_lidar_pts_per_ins'][ins_id][iter_i]
                # num_cams = len(cnt_lidar_pts_per_ins_per_iter)
                # cnt_lidar_pts_per_ins_per_iter = np.repeat(cnt_lidar_pts_per_ins_per_iter.reshape((1, num_cams)), num_cams, axis=0)
                psnr_eval_mat = cross_eval_result['psnr_eval_mat_per_ins'][ins_id][iter_i]
                depth_eval_mat = cross_eval_result['depth_eval_mat_per_ins'][ins_id][iter_i]

                # discard the original view eval results on the diagonal of the cross matrix
                r_vec, c_vec = np.where(~np.eye(num_cams, dtype=bool))
                psnr_cross_eval_all[iter_i] += (psnr_eval_mat[r_vec, c_vec]).tolist()
                depth_cross_eval_all[iter_i] += (depth_eval_mat[r_vec, c_vec]).tolist()
                # cnt_lidar_cross_eval_all[iter_i] += (cnt_lidar_pts_per_ins_per_iter[r_vec, c_vec]).tolist()

        psnr_cross_eval_mean = np.zeros(n_iters)
        depth_cross_eval_mean = np.zeros(n_iters)
        for iter_i in range(0, n_iters):
            psnr_cross_eval_mean[iter_i] = np.array(psnr_cross_eval_all[iter_i]).mean()
            # depth_cross_eval_mean[iter_i] = np.sum(np.array(depth_cross_eval_all[iter_i]) *
            #                                        np.array(cnt_lidar_cross_eval_all[iter_i])) / \
            #                                 np.sum(cnt_lidar_cross_eval_all[iter_i])
            depth_cross_eval_mean[iter_i] = np.mean(np.array(depth_cross_eval_all[iter_i]))
        if len(axes.shape) > 1:
            axe = axes[0, 0]
        else:
            axe = axes[0]
        axe.plot(code_save_iters, psnr_cross_eval_mean, f'{color}s--', linewidth=2)
        print(f'    psnr cross-view: {np.round(psnr_cross_eval_mean, 2)}')

        if len(axes.shape) > 1:
            axe = axes[0, 1]
        else:
            axe = axes[1]
        axe.plot(code_save_iters, depth_cross_eval_mean, f'{color}s--', linewidth=2)
        print(f'    depth err cross-view: {np.round(depth_cross_eval_mean, 2)}')

    return lines


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
                Modified from NUSC

    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def view_points_batch(points: torch.tensor, view: torch.tensor, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[1] <= 4
    assert view.shape[2] <= 4
    assert points.shape[1] == 3

    device = points.get_device()
    if device < 0:
        device = 'cpu'
    bsize = view.shape[0]
    viewpad = np.eye(4).reshape((1, 4, 4)).astype(np.float32)
    viewpad = torch.from_numpy(np.repeat(viewpad, bsize, axis=0)).to(device)
    viewpad[:, :view.shape[1], :view.shape[2]] = view

    nbr_points = points.shape[2]

    # Do operation in homogenous coordinates.
    points = torch.cat([points, torch.ones((bsize, 1, nbr_points), dtype=torch.float32).to(device)], dim=1)
    points = torch.matmul(viewpad, points)
    points = points[:, :3, :]

    if normalize:
        points = points / points[:, 2:3, :]

    return points


def corners_of_box(obj_pose, wlh, is_kitti=False):
    """
            Modified from NUSC

    Returns the bounding box corners.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = wlh

    if is_kitti:
        # the order (identity) need to follow nusc
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = h / 2 * np.array([-2, -2, 0, 0, -2, -2, 0, 0])
        z_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    else:
        # 3D bounding box corners. (Convention: x forward, y left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(obj_pose[:, :3], corners)

    # Translate
    x, y, z = obj_pose[:, 3]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def corners_of_box_batch(obj_pose_batch, wlh_batch, is_kitti=False, scale=1.0):
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    device = wlh_batch.get_device()
    if device < 0:
        device = 'cpu'
    w = wlh_batch[:, 0]
    l = wlh_batch[:, 1]
    h = wlh_batch[:, 2]

    if is_kitti:
        # the order (identity) need to follow nusc
        x_corners = l.view((-1, 1, 1)) / 2 * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1]).view((1, 1, -1)).to(device) * scale
        y_corners = h.view((-1, 1, 1)) / 2 * torch.tensor([-2, -2, 0, 0, -2, -2, 0, 0]).view((1, 1, -1)).to(device) * scale
        z_corners = w.view((-1, 1, 1)) / 2 * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1]).view((1, 1, -1)).to(device) * scale
    else:
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l.view((-1, 1, 1)) / 2 * torch.tensor([1, 1, 1, 1, -1, -1, -1, -1]).view((1, 1, -1)).to(device) * scale
        y_corners = w.view((-1, 1, 1)) / 2 * torch.tensor([1, -1, -1, 1, 1, -1, -1, 1]).view((1, 1, -1)).to(device) * scale
        z_corners = h.view((-1, 1, 1)) / 2 * torch.tensor([1, 1, -1, -1, 1, 1, -1, -1]).view((1, 1, -1)).to(device) * scale
    corners = torch.cat((x_corners, y_corners, z_corners), dim=1)

    # Rotate
    corners = torch.matmul(obj_pose_batch[:, :, :3], corners)

    # Translate
    x = obj_pose_batch[:, 0, 3]
    y = obj_pose_batch[:, 1, 3]
    z = obj_pose_batch[:, 2, 3]
    corners[:, 0, :] = corners[:, 0, :] + x.unsqueeze(-1)
    corners[:, 1, :] = corners[:, 1, :] + y.unsqueeze(-1)
    corners[:, 2, :] = corners[:, 2, :] + z.unsqueeze(-1)

    return corners


def pts_in_box_3d(pts_3d, corners_3d, keep_top_portion=1.0):
    """
        Extract lidar points associated within the annotation 3D box
        Both pts_3d and corners_3d (nusc def) live in the same coordinate frame
        keep_ratio is the top portion to be kept

        return the indices
    """
    v1 = (corners_3d[:, 1:2] - corners_3d[:, 0:1])
    v2 = (corners_3d[:, 3:4] - corners_3d[:, 0:1]) * keep_top_portion
    v3 = (corners_3d[:, 4:5] - corners_3d[:, 0:1])
    v_test = pts_3d - corners_3d[:, 0:1]

    proj_1 = np.matmul(v1.T, v_test)
    proj_2 = np.matmul(v2.T, v_test)
    proj_3 = np.matmul(v3.T, v_test)

    subset1 = np.logical_and(proj_1 > 0,  proj_1 < np.matmul(v1.T, v1))
    subset2 = np.logical_and(proj_2 > 0,  proj_2 < np.matmul(v2.T, v2))
    subset3 = np.logical_and(proj_3 > 0,  proj_3 < np.matmul(v3.T, v3))

    subset_final = np.logical_and(subset1, np.logical_and(subset2, subset3))
    return np.squeeze(subset_final)


def normalize_by_roi(pts_batch, roi_batch, need_square=True):
    """
        pts_batch: N x 2 x npts
        roi_batch: N x 4 (xmin, ymin, xmax, ymax)
        if need_square, assume the longer side of roi is used for normalization
    """
    w_batch = roi_batch[:, 2] - roi_batch[:, 0]
    h_batch = roi_batch[:, 3] - roi_batch[:, 1]
    cx_batch = (roi_batch[:, 2] + roi_batch[:, 0]) / 2
    cy_batch = (roi_batch[:, 3] + roi_batch[:, 1]) / 2

    pts_norm = pts_batch.clone()
    pts_norm[:, 0, :] -= cx_batch.unsqueeze(-1)
    pts_norm[:, 1, :] -= cy_batch.unsqueeze(-1)

    if need_square:
        dim_batch = torch.maximum(w_batch, h_batch)
        pts_norm /= dim_batch.view((-1, 1, 1))
    else:
        pts_norm[:, 0, :] /= w_batch.unsqueeze(-1)
        pts_norm[:, 1, :] /= h_batch.unsqueeze(-1)
        dim_batch = None
    return pts_norm, dim_batch


def render_box(
        im: np.ndarray,
        corners_2d: np.ndarray,
        colors: tuple = ('b', 'r', 'k'),
        linewidth: float = 2):

    """
        Modified from NUSC
    """
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                 (int(corners_2d.T[i][0]), int(corners_2d.T[i][1])),
                 (int(corners_2d.T[i + 4][0]), int(corners_2d.T[i + 4][1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners_2d.T[:4], colors[0][::-1])
    draw_rect(corners_2d.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners_2d.T[2:4], axis=0)
    center_bottom = np.mean(corners_2d.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    return im


def draw_boxes_train(generated_img, src_uv_vis_ii, pred_uv_end_ii, pred_uv_direct_ii, roi, out_uv):
    src_uv_vis_ii[0, :] -= roi[0].item()
    src_uv_vis_ii[1, :] -= roi[1].item()
    pred_uv_end_ii[0, :] -= roi[0].item()
    pred_uv_end_ii[1, :] -= roi[1].item()

    r_c = np.array([0, 0, 1]).astype(np.float64)
    g_c = np.array([0, 1, 0]).astype(np.float64)
    generated_img = render_box(generated_img, src_uv_vis_ii, colors=(r_c, r_c, r_c))
    generated_img = render_box(generated_img, pred_uv_end_ii, colors=(g_c, g_c, g_c))

    if out_uv:
        pred_uv_direct_ii[0, :] -= roi[0].item()
        pred_uv_direct_ii[1, :] -= roi[1].item()

        b_c = np.array([1, 0, 0]).astype(np.float64)
        generated_img = render_box(generated_img, pred_uv_direct_ii, colors=(b_c, b_c, b_c))
    return generated_img


def get_random_pose(tgt_pose, K, roi, yaw_lim=np.pi/2, angle_lim=np.pi/9, trans_lim=0.3, depth_lim=0.3):
    """
        Used in generating randomized src pose for training dl-based pose refiner
        trans in u, v, Z space,
        Rotation large yaw, limited pitch, roll

        Currently only apply to nusc, no adaption to kitti
    """
    # use padded K for both nusc and kitti use
    viewpad = np.eye(4)
    viewpad[:K.shape[0], :K.shape[1]] = K

    tgt_pose_uv = view_points(tgt_pose[:, 3:], K, normalize=True)
    roi_w = (roi[2] - roi[0]).item()
    roi_h = (roi[3] - roi[1]).item()
    v_x = random.uniform(-roi_w * trans_lim, roi_w * trans_lim)
    v_y = random.uniform(-roi_h * trans_lim, roi_h * trans_lim)
    v_z = random.uniform(1. - depth_lim, 1. + depth_lim)
    src_obj_u = tgt_pose_uv[0, 0] + v_x
    src_obj_v = tgt_pose_uv[1, 0] + v_y
    src_obj_Z = tgt_pose[2, 3].item() * v_z
    # T_src = np.linalg.inv(K) @ np.array([src_obj_u, src_obj_v, 1]).reshape((3, 1)) * src_obj_Z
    T_src = np.linalg.inv(viewpad) @ np.array([src_obj_u * src_obj_Z, src_obj_v * src_obj_Z, src_obj_Z, 1]).reshape(
        (4, 1))
    T_src = T_src[:3]
    yaw_err = random.uniform(-yaw_lim, yaw_lim)
    R_yaw = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                        [np.sin(yaw_err), np.cos(yaw_err), 0.],
                        [0., 0., 1.]]).astype(np.float32)
    rotvec_rand = [random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim)]
    R_rand = R.from_rotvec(rotvec_rand).as_matrix()

    R_src = tgt_pose[:, :3] @ R_rand @ R_yaw
    src_pose = np.concatenate([R_src, T_src], axis=1)

    return src_pose


def get_random_pose2(K, roi, yaw_lim=np.pi, angle_lim=np.pi / 9, trans_lim=0.4, depth_fix=20, is_kitti=False):
    """
        Used in generating randomized src pose for test
        trans in u, v, Z space,
        Rotation all range, limited pitch, roll

        By default the object frame def follows nusc. Adapt to kitti in case

    """
    # use padded K for both nusc and kitti use
    viewpad = np.eye(4)
    viewpad[:K.shape[0], :K.shape[1]] = K

    roi_cx = (roi[2] + roi[0]) / 2
    roi_cy = (roi[3] + roi[1]) / 2
    roi_w = (roi[2] - roi[0])
    roi_h = (roi[3] - roi[1])
    v_x = random.uniform(-roi_w * trans_lim, roi_w * trans_lim)
    v_y = random.uniform(-roi_h * trans_lim, roi_h * trans_lim)
    src_obj_u = roi_cx + v_x
    src_obj_v = roi_cy + v_y
    src_obj_Z = depth_fix

    # T_src = np.linalg.inv(K) @ np.array([src_obj_u, src_obj_v, 1]).reshape((3, 1)) * src_obj_Z
    T_src = np.linalg.inv(viewpad) @ np.array([src_obj_u * src_obj_Z, src_obj_v * src_obj_Z, src_obj_Z, 1]).reshape((4, 1))
    T_src = T_src[:3]

    yaw_err = random.uniform(-yaw_lim, yaw_lim)

    if is_kitti:  # kitti object frame x-front, y-down, z-left
        R_unit = np.array([[0., 0., -1.],
                           [0., 1., 0.],
                           [1., 0., 0.]])
        R_yaw = np.array([[np.cos(yaw_err), 0., np.sin(yaw_err)],
                          [0., 1., 0.],
                          [-np.sin(yaw_err), 0., np.cos(yaw_err)]]).astype(np.float32)
    else:  # nusc object frame x-front, y-left, z-up
        R_unit = np.array([[0., -1.,  0.],
                           [0.,  0., -1.],
                           [1.,  0.,  0.]])
        R_yaw = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                            [np.sin(yaw_err), np.cos(yaw_err), 0.],
                            [0., 0., 1.]]).astype(np.float32)
    rotvec_rand = [random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim)]
    R_rand = R.from_rotvec(rotvec_rand).as_matrix()

    R_src = R_unit @ R_rand @ R_yaw  # this formular augment under object frame, more general without assuming camera pose
    src_pose = np.concatenate([R_src, T_src], axis=1)

    return src_pose


def obj_pose_kitti2nusc(obj_pose_src, obj_h):
    """
        Operate at batch level
    """
    pose_R = obj_pose_src[:, :, :3]
    pose_T = obj_pose_src[:, :, 3:]
    pose_T[:, 1, 0] -= (obj_h/2)
    R_x = np.array([[1., 0., 0.],
                    [0., 0., -1.],
                    [0., 1., 0.]]).astype(np.float32)
    R_x = torch.from_numpy(R_x).unsqueeze(0).repeat(obj_pose_src.shape[0], 1, 1)
    pose_R = torch.matmul(pose_R, R_x)
    return torch.cat([pose_R, pose_T], dim=-1)


def obj_pose_nuse2kitti(obj_pose_src, obj_h):
    """
        Operate at batch level
    """
    pose_R = obj_pose_src[:, :, :3]
    pose_T = obj_pose_src[:, :, 3:]
    pose_T[:, 1:, 0] += (obj_h/2)
    R_x = np.array([[1., 0., 0.],
                    [0., 0., 1.],
                    [0., -1., 0.]]).astype(np.float32)
    R_x = torch.from_numpy(R_x).unsqueeze(0).repeat(obj_pose_src.shape[0], 1, 1)
    pose_R = torch.matmul(pose_R, R_x)
    return torch.cat([pose_R, pose_T], dim=-1)


def roi_coord_trans(x_vec, y_vec, roi_src, im_sz_tgt):
    roi_w = roi_src[2] - roi_src[0]
    roi_h = roi_src[3] - roi_src[1]
    x_vec_new = (x_vec - roi_w / 2) / roi_w * im_sz_tgt + im_sz_tgt / 2
    y_vec_new = (y_vec - roi_h / 2) / roi_h * im_sz_tgt + im_sz_tgt / 2
    return x_vec_new, y_vec_new


def roi_process(roi, H=None, W=None, roi_margin=0, sq_pad=False):
    """
        roi is in tensor [xmin, ymin, xmax, ymax]
    """
    roi_new = roi.clone()
    roi_new[0:2] -= roi_margin
    roi_new[2:4] += roi_margin

    if sq_pad:  # pad the shorter side
        center_x = (roi_new[0] + roi_new[2]) / 2
        center_y = (roi_new[1] + roi_new[3]) / 2
        sz = np.maximum(roi_new[2] - roi_new[0], roi_new[3] - roi_new[1])
        roi_new[0] = center_x - sz / 2
        roi_new[2] = center_x + sz / 2
        roi_new[1] = center_y - sz / 2
        roi_new[3] = center_y + sz / 2

    # cut the in-image part (not square for truncated cases)
    if H is not None and W is not None:
        roi_new[0:2] = torch.maximum(roi_new[0:2], torch.as_tensor(0))
        roi_new[2] = torch.minimum(roi_new[2], torch.as_tensor(W - 1))
        roi_new[3] = torch.minimum(roi_new[3], torch.as_tensor(H - 1))

    return roi_new


def roi_resize(roi, ratio=1.0):
    min_x, min_y, max_x, max_y = roi
    # enlarge pred_box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    box_w = max_x - min_x
    box_h = max_y - min_y
    min_x = center_x - box_w / 2 * ratio
    max_x = center_x + box_w / 2 * ratio
    min_y = center_y - box_h / 2 * ratio
    max_y = center_y + box_h / 2 * ratio
    roi = [min_x, min_y, max_x, max_y]
    return roi


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img