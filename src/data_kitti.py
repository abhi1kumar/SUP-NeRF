import random
import numpy as np
import torch
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

from utils import pts_in_box_3d, get_random_pose2, corners_of_box, render_box, view_points, roi_resize
from data.KITTI.kitti_object_vis.kitti_object import kitti_object, show_lidar_on_image, show_image_with_boxes, get_lidar_in_image_fov
from data.KITTI.kitti_object_vis.kitti_util import compute_box_3d, draw_projected_box3d

# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L14
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def ins2vis(masks, tgt_ins_id=None):
    masks = np.asarray(masks)
    img = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)

    for ins_id, mask in enumerate(masks):
        yy, xx = np.where(mask > 0)
        img[yy, xx, :] = _COLORS[ins_id % _COLORS.shape[0]]
        if tgt_ins_id is not None and ins_id == tgt_ins_id:
            img[yy, xx, :] = 1
    return img


def get_mask_occ_from_ins(masks, tgt_ins_id):
    """
        Prepare occupancy mask:
            target object: 1
            background: -1 (not likely to occlude the object)
            occluded the instance: 0 (seems only able to reason the occlusion by foreground)
    """
    tgt_mask = masks[tgt_ins_id]
    mask_occ = np.zeros_like(tgt_mask).astype(np.int32)
    mask_union = np.sum(np.asarray(masks), axis=0)

    mask_occ[mask_union == 0] = -1
    mask_occ[tgt_mask > 0] = 1
    return mask_occ


def get_associate_box_3d(objects, tgt_mask, tgt_cat):
    """
        Objects are loaded third-party detector results
    """
    y, x = np.where(tgt_mask > 0)
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    tgt_box_area = (max_x - min_x) * (max_y - min_y)
    max_id = -1
    box_iou = 0.0

    for ii, obj in enumerate(objects):
        if obj.type != tgt_cat:
            continue
        min_x2, min_y2, max_x2, max_y2 = obj.box2d
        x_left = max(min_x, min_x2)
        y_top = max(min_y, min_y2)
        x_right = min(max_x, max_x2)
        y_bottom = min(max_y, max_y2)

        if x_right < x_left or y_bottom < y_top:
            box_iou_i = 0.0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            union = tgt_box_area + (max_x2 - min_x2) * (max_y2 - min_y2) - intersection
            box_iou_i = intersection / union

        if box_iou_i > box_iou:
            max_id = ii
            box_iou = box_iou_i
    return max_id, box_iou


def get_tgt_ins_from_masksrcnn_v2(preds, masks, tgt_cat, tgt_box, lidar_pts_im):
    """
        Use cnt to locate the target segment
        Assumptions:
            preds and masks are predicted from maskrcnn
            The prediction do not perfect match tgt_box
            lidar points are only associated with the target annotation
    """
    # locate the detections matched the target category
    indices = [idx for idx, label in enumerate(preds['labels']) if tgt_cat in label]
    if len(indices) == 0 or lidar_pts_im.shape[1] == 0:
        return None, 0, 0., 0., 0

    boxes = np.asarray(preds['boxes'])[indices]
    masks = np.asarray(masks)[indices]/255
    # calculate box ious between predicted boxed and tgt box
    lidar_reads = masks[:, lidar_pts_im[1, :].astype(np.int32), lidar_pts_im[0, :].astype(np.int32)]
    lidar_cnts = np.sum(lidar_reads, axis=1)
    max_id = np.argmax(lidar_cnts)
    lidar_cnt = lidar_cnts[max_id]

    out_ins_id = indices[max_id]
    out_mask = masks[max_id]
    out_ins_area = np.sum((out_mask > 0).astype(np.int32))
    out_box = boxes[max_id]
    out_box_area = (out_box[2] - out_box[0]) * (out_box[3] - out_box[1])
    # the ratio of predicted mask under predicted box
    area_ratio = float(out_ins_area) / out_box_area
    # the iou between the target box and pred box
    min_x, min_y, max_x, max_y = tgt_box
    tgt_box_area = (max_x - min_x) * (max_y - min_y)
    min_x2, min_y2, max_x2, max_y2 = out_box
    x_left = max(min_x, min_x2)
    y_top = max(min_y, min_y2)
    x_right = min(max_x, max_x2)
    y_bottom = min(max_y, max_y2)
    if x_right < x_left or y_bottom < y_top:
        box_iou = 0.0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = tgt_box_area + (max_x2 - min_x2) * (max_y2 - min_y2) - intersection
        box_iou = intersection / union

    return out_ins_id, out_ins_area, area_ratio, box_iou, lidar_cnt


class KittiData:
    def __init__(self, hpams,
                 kitti_data_dir,
                 split='val',
                 out_gt_depth=True,
                 add_pose_err=0,
                 init_rot_err=0.2,
                 init_trans_err=0.01,
                 rand_angle_lim=0,
                 pred_box2d=False,
                 box2d_rz_ratio=1.2,
                 debug=False,
                 ):
        """
            Provide camera input and label per annotation per instance for the target category
            Object 'Instance' here respects the definition from KITTI dataset.
            Each instance of object does not go beyond a single scene.
            Each instance contains multiple annotations from different timestamps
            Each annotation could be projected to multiple camera inputs at the same timestamp.
        """
        self.kitti_cat = hpams['dataset']['kitti_cat']
        self.seg_cat = hpams['dataset']['seg_cat']
        self.divisor = hpams['dataset']['divisor']
        self.box_iou_th = hpams['dataset']['box_iou_th']
        self.max_dist = hpams['dataset']['max_dist']
        self.min_depth = hpams['dataset']['min_depth']
        self.min_lidar_cnt = hpams['dataset']['min_lidar_cnt']
        self.mask_pixels = hpams['dataset']['mask_pixels']
        self.split_dir = hpams['dataset']['split_dir']
        self.out_gt_depth = out_gt_depth
        self.debug = debug
        self.pred_box2d = pred_box2d
        self.box2d_rz_ratio = box2d_rz_ratio

        self.kitti_data_dir = kitti_data_dir
        self.seg_type = 'instance'
        if split != 'test':
            self.kitti = kitti_object(self.kitti_data_dir, split='training', args=None)
            self.kitti_seg_dir = os.path.join(self.kitti_data_dir, 'training/pred_instance')
        else:
            self.kitti = kitti_object(self.kitti_data_dir, split='testing', args=None)
            self.kitti_seg_dir = os.path.join(self.kitti_data_dir, 'testing/pred_instance')
        self.all_valid_samples = []  # (anntoken, cam) pairs
        self.sample_attr = {}  # save attributes of valid samples for fast retrieval at run time

        # Prepare index file for the valid samples for later efficient batch preparation
        subset_index_file = 'data/KITTI/kitti.' + split + '.' + self.kitti_cat + '.json'
        if os.path.exists(subset_index_file):
            kitti_subset = json.load(open(subset_index_file))
            if kitti_subset['box_iou_th'] != self.box_iou_th or kitti_subset['max_dist'] != self.max_dist or kitti_subset[
                'mask_pixels'] != self.mask_pixels or kitti_subset['min_lidar_cnt'] != self.min_lidar_cnt or kitti_subset[
                'seg_type'] != self.seg_type or kitti_subset['min_depth'] != self.min_depth:
                print('Different dataset config found! Re-preprocess the dataset to prepare indices of valid samples...')
                self.preprocess_dataset(split, subset_index_file)
            else:
                self.all_valid_samples = kitti_subset['all_valid_samples']
                self.sample_attr = kitti_subset['sample_attr']
                print('Loaded existing index file for valid samples.')
        else:
            print('No existing index file found! Preprocess the dataset to prepare indices of valid samples...')
            self.preprocess_dataset(split, subset_index_file)

        self.lenids = len(self.all_valid_samples)
        print(f'{self.lenids} annotations in {self.kitti_cat} category are included in dataloader.')

        # for adding error to pose
        self.add_pose_err = add_pose_err
        self.init_rot_err = init_rot_err
        self.init_trans_err = init_trans_err
        self.rand_angle_lim = rand_angle_lim

    def preprocess_dataset(self, split, subset_file):
        """
            Go through the full dataset once to save the valid indices. Save the index file for later direct refer.
        """
        split_file = os.path.join(self.split_dir, split + '.txt')
        with open(split_file) as file:
            data_ids = [line.rstrip() for line in file]

        # retrieve all the target instance
        for data_idx in tqdm(data_ids):
            # load data and labels
            pc_velo = self.kitti.get_lidar(int(data_idx), np.float32, 4)[:, 0:4]
            calib = self.kitti.get_calibration(int(data_idx))
            img = self.kitti.get_image(int(data_idx))
            objects = self.kitti.get_label_objects(int(data_idx))
            img_height, img_width, _ = img.shape
            K = calib.P[:, :3]
            # if calib.P[0, 3] != 0 or calib.P[1, 3] != 0:
            #     print('found irregular camera intrinsic')
            #     print(calib.P)

            # get lidar projections in image FOV
            imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True
            )
            lidar_pts_im = pts_2d[fov_inds, :]
            lidar_pts_im = lidar_pts_im.transpose()
            imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
            imgfov_pc_rect = imgfov_pc_rect.transpose()
            lider_pts_depth = imgfov_pc_rect[2, :]

            # img = show_lidar_on_image(pc_velo[:, 0:3], img, calib, self.img_w, self.img_h)
            # _, img = show_image_with_boxes(img, objects, calib, True, depth=None)

            # # iterate through each detection label, saving lidar can save run-time loading a lot
            for obj_idx, obj in enumerate(objects):
                # box here is in sensor coordinate system
                if obj.type != self.kitti_cat:
                    continue

                box_2d = obj.box2d
                Ry = np.array([[np.cos(obj.ry), 0., np.sin(obj.ry)],
                               [0., 1., 0.],
                               [-np.sin(obj.ry), 0., np.cos(obj.ry)]]).astype(np.float32)
                # kitti's object frame is defined in a wield way x-front, y-down, z left
                R_obj = Ry
                # convert the extra column in K back to object Translation
                T_obj = np.asarray(obj.t).reshape(3, 1) + np.linalg.inv(K) @ calib.P[:, 3:]
                obj_pose = np.concatenate([R_obj, T_obj], axis=1)
                wlh = np.array([obj.w, obj.l, obj.h])

                # ATTENTION: kitti obj location is defined on the ground, but nusc as box center
                corners_3d = corners_of_box(obj_pose, wlh, is_kitti=True)
                pts_ann_indices = pts_in_box_3d(imgfov_pc_rect, corners_3d, keep_top_portion=0.9)
                lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]
                lider_pts_depth_ann = lider_pts_depth[pts_ann_indices]

                # process mask files
                json_file = os.path.join(self.kitti_seg_dir,
                                         data_idx + '.json')
                preds = json.load(open(json_file))
                ins_masks = []
                for box_id in range(0, len(preds['boxes'])):
                    mask_file = os.path.join(self.kitti_seg_dir,
                                             data_idx + f'_{box_id}.png')
                    mask = np.asarray(Image.open(mask_file))
                    ins_masks.append(mask)

                tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou, lidar_cnt = get_tgt_ins_from_masksrcnn_v2(
                    preds,
                    ins_masks,
                    self.seg_cat,
                    box_2d,
                    lidar_pts_im_ann)

                # save the qualified sample index for later direct use
                is_valid = False
                if tgt_ins_id is not None and tgt_ins_cnt > self.mask_pixels and box_iou > self.box_iou_th and \
                        area_ratio > self.box_iou_th and np.linalg.norm(T_obj) < self.max_dist and \
                        T_obj[2] > self.min_depth and lidar_cnt >= self.min_lidar_cnt and \
                        obj.occlusion < 3 and obj.truncation == 0:
                    self.all_valid_samples.append([data_idx, str(obj_idx)])
                    # ATTENTION: the samples record here are not all valid
                    if data_idx not in self.sample_attr.keys():
                        self.sample_attr[data_idx] = {}
                    if obj_idx not in self.sample_attr[data_idx].keys():
                        self.sample_attr[data_idx][str(obj_idx)] = {'seg_id': tgt_ins_id, 'lidar_cnt': lidar_cnt}
                    is_valid = True

                if self.debug:
                    print(
                        f'        tgt instance id: {tgt_ins_id}, '
                        f'lidar pts cnt: {lidar_cnt} ')

                    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

                    img2 = np.copy(img)
                    pred_uv = view_points(
                        corners_3d,
                        K, normalize=True)
                    c = np.array([0, 255, 0]).astype(np.float)
                    img2 = render_box(img2, pred_uv, colors=(c, c, c))
                    # box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
                    # img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
                    axes[0].imshow(img2)
                    axes[0].set_title(f'Camera wt box3d (valid: {is_valid})')
                    axes[0].axis('off')
                    axes[0].set_aspect('equal')

                    seg_vis = ins2vis(ins_masks)
                    axes[1].imshow(seg_vis)
                    axes[1].set_title(f'pred instance (valid: {is_valid})')
                    axes[1].axis('off')
                    axes[1].set_aspect('equal')

                    min_x, min_y, max_x, max_y = box_2d
                    rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                             linewidth=2, edgecolor='y', facecolor='none')
                    axes[1].add_patch(rect)

                    # axes[0].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=2)
                    axes[1].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=1)
                    plt.tight_layout()
                    plt.show()

        # save into json file for quick load next time
        kitti_subset = {}
        kitti_subset['all_valid_samples'] = self.all_valid_samples
        kitti_subset['sample_attr'] = self.sample_attr
        kitti_subset['box_iou_th'] = self.box_iou_th
        kitti_subset['max_dist'] = self.max_dist
        kitti_subset['min_depth'] = self.min_depth
        kitti_subset['mask_pixels'] = self.mask_pixels
        kitti_subset['min_lidar_cnt'] = self.min_lidar_cnt
        kitti_subset['seg_type'] = self.seg_type
        json.dump(kitti_subset, open(subset_file, 'w'), indent=4)

    def __len__(self):
        return self.lenids

    def __getitem__(self, idx):
        sample_data = {}
        data_idx, obj_idx = self.all_valid_samples[idx]
        if self.debug:
            print(f'frame: {data_idx}')

        # load data and labels
        pc_velo = self.kitti.get_lidar(int(data_idx), np.float32, 4)[:, 0:4]
        calib = self.kitti.get_calibration(int(data_idx))
        img = self.kitti.get_image(int(data_idx))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects = self.kitti.get_label_objects(int(data_idx))
        img_height, img_width, _ = img.shape
        K = calib.P[:, :3]
        # if calib.P[0, 3] != 0 or calib.P[1, 3] != 0:
        #     print('found irregular camera intrinsic')
        #     print(calib.P)
        obj = objects[int(obj_idx)]

        box_2d = obj.box2d
        Ry = np.array([[np.cos(obj.ry), 0., np.sin(obj.ry)],
                       [0., 1., 0.],
                       [-np.sin(obj.ry), 0., np.cos(obj.ry)]]).astype(np.float32)
        # kitti's object frame is defined in a wield way x-front, y-down, z left
        R_obj = Ry
        # convert the extra column in K back to object Translation
        T_obj = np.asarray(obj.t).reshape(3, 1) + np.linalg.inv(K) @ calib.P[:, 3:]
        obj_pose = np.concatenate([R_obj, T_obj], axis=1)
        # Compute camera pose in object frame = c2o transformation matrix
        R_c2o = R_obj.transpose()
        t_c2o = - R_c2o @ T_obj
        cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)
        wlh = np.array([obj.w, obj.l, obj.h])
        corners_3d = corners_of_box(obj_pose, wlh, is_kitti=True)

        # process mask files
        json_file = os.path.join(self.kitti_seg_dir,
                                 data_idx + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        for box_id in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.kitti_seg_dir,
                                     data_idx + f'_{box_id}.png')
            mask = np.asarray(Image.open(mask_file))
            ins_masks.append(mask)

        tgt_ins_id = self.sample_attr[data_idx][obj_idx]['seg_id']
        mask_occ = get_mask_occ_from_ins(ins_masks, tgt_ins_id)
        lidar_cnt = self.sample_attr[data_idx][obj_idx]['lidar_cnt']
        if self.pred_box2d:
            box_2d = preds['boxes'][tgt_ins_id]
            # enlarge pred_box
            box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)

        if self.add_pose_err == 1:
            # only consider yaw error and distance error
            # yaw_err = random.uniform(-self.max_rot_pert, self.max_rot_pert)
            yaw_err = random.choice([1., -1.]) * self.init_rot_err
            rot_err = np.array([[np.cos(yaw_err), 0., np.sin(yaw_err)],
                                [0., 1., 0.],
                                [-np.sin(yaw_err), 0., np.cos(yaw_err)]]).astype(np.float32)
            # trans_err_ratio = random.uniform(1.0-self.max_t_pert, 1.0+self.max_t_pert)
            # TODO: applying trans err seems problematic
            trans_err_ratio = 1. + random.choice([1., -1.]) * self.init_trans_err
            obj_center_w_err = T_obj * trans_err_ratio
            obj_orientation_w_err = R_obj @ rot_err  # rot error need to right --> to model points
            obj_pose_w_err = np.concatenate([obj_orientation_w_err, obj_center_w_err], axis=1)
            R_c2o_w_err = obj_orientation_w_err.transpose()
            t_c2o_w_err = -R_c2o_w_err @ obj_center_w_err
            cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)

            sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose_w_err.astype(np.float32))
            sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose_w_err.astype(np.float32))
        elif self.add_pose_err >= 2:
            # adapted to  kitti version
            obj_pose_w_err = get_random_pose2(K,
                                              np.asarray(box_2d).astype(np.int32),
                                              yaw_lim=np.pi, angle_lim=self.rand_angle_lim,  # kitti only has yaw rotation
                                              trans_lim=0.3, depth_fix=20, is_kitti=True)
            R_c2o_w_err = obj_pose_w_err[:3, :3].T
            t_c2o_w_err = -R_c2o_w_err @ obj_pose_w_err[:3, 3:]
            cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)
            sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose_w_err.astype(np.float32))
            sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose_w_err.astype(np.float32))
        else:
            sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose.astype(np.float32))
            sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose.astype(np.float32))

        # Associate with third-party detection results (Separate condition in case no association exists)
        if self.add_pose_err == 3:
            objects_pred = self.kitti.get_pred_objects(int(data_idx))
            asso_obx_id, box_iou = get_associate_box_3d(objects_pred, ins_masks[tgt_ins_id], self.kitti_cat)
            if asso_obx_id >= 0 and box_iou > 0:
                obj_pred = objects_pred[asso_obx_id]
                Ry = np.array([[np.cos(obj_pred.ry), 0., np.sin(obj_pred.ry)],
                               [0., 1., 0.],
                               [-np.sin(obj_pred.ry), 0., np.cos(obj_pred.ry)]]).astype(np.float32)
                # kitti's object frame is defined in a wield way x-front, y-down, z left
                R_obj = Ry
                # convert the extra column in K back to object Translation
                T_obj = np.asarray(obj_pred.t).reshape(3, 1) + np.linalg.inv(K) @ calib.P[:, 3:]
                obj_pose_w_err = np.concatenate([R_obj, T_obj], axis=1)
                # Compute camera pose in object frame = c2o transformation matrix
                R_c2o = R_obj.transpose()
                t_c2o = - R_c2o @ T_obj
                cam_pose_w_err = np.concatenate([R_c2o, t_c2o], axis=1)
                # wlh = np.array([obj.w, obj.l, obj.h])
                # corners_3d = corners_of_box(obj_pose, wlh, is_kitti=True)
                sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose_w_err.astype(np.float32))
                sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose_w_err.astype(np.float32))

        if self.out_gt_depth or self.debug:
            # get lidar projections in image FOV
            imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True
            )
            lidar_pts_im = pts_2d[fov_inds, :]
            lidar_pts_im = lidar_pts_im.transpose()
            imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
            imgfov_pc_rect = imgfov_pc_rect.transpose()
            lider_pts_depth = imgfov_pc_rect[2, :]

            # ATTENTION: kitti obj location is defined on the ground, but nusc as box center
            pts_ann_indices = pts_in_box_3d(imgfov_pc_rect, corners_3d, keep_top_portion=0.9)
            lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]
            lider_pts_depth_ann = lider_pts_depth[pts_ann_indices]

            depth_map = np.zeros(img.shape[:2]).astype(np.float32)
            depth_map[lidar_pts_im_ann[1, :].astype(np.int32), lidar_pts_im_ann[0, :].astype(np.int32)] = lider_pts_depth_ann
            sample_data['depth_maps'] = torch.from_numpy(depth_map.astype(np.float32))

        sample_data['imgs'] = torch.from_numpy(img.astype(np.float32)/255.)
        sample_data['masks_occ'] = torch.from_numpy(mask_occ.astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(box_2d).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(K.astype(np.float32))
        sample_data['cam_poses'] = torch.from_numpy(np.asarray(cam_pose).astype(np.float32))
        sample_data['obj_poses'] = torch.from_numpy(np.asarray(obj_pose).astype(np.float32))
        sample_data['data_idx'] = data_idx
        sample_data['obj_idx'] = obj_idx
        sample_data['wlh'] = torch.tensor(wlh, dtype=torch.float32)
        sample_data['occlusion'] = obj.occlusion

        if self.debug:
            print(
                f'        tgt instance id: {tgt_ins_id}, '
                f'lidar pts cnt: {lidar_cnt} ')

            fig, axes = plt.subplots(2, 1, figsize=(8, 8))

            # draw object box on the image
            img2 = np.copy(img)
            pred_uv = view_points(
                corners_3d,
                K, normalize=True)
            c = np.array([0, 255, 0]).astype(np.float)
            img2 = render_box(img2, pred_uv, colors=(c, c, c))
            if self.add_pose_err > 0:
                corners_3d_w_err = corners_of_box(obj_pose_w_err, wlh, is_kitti=True)
                pred_uv_w_err = view_points(
                    corners_3d_w_err,
                    K, normalize=True)
                c = np.array([255, 0, 0]).astype(np.float)
                img2 = render_box(img2, pred_uv_w_err, colors=(c, c, c))
            # box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
            # img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
            axes[0].imshow(img2)
            axes[0].set_title('Camera wt box3d')
            axes[0].axis('off')
            axes[0].set_aspect('equal')

            seg_vis = ins2vis(ins_masks, tgt_ins_id)
            axes[1].imshow(seg_vis)
            axes[1].set_title('pred instance')
            axes[1].axis('off')
            axes[1].set_aspect('equal')

            min_x, min_y, max_x, max_y = box_2d
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                     linewidth=2, edgecolor='y', facecolor='none')
            axes[1].add_patch(rect)

            # axes[0].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=2)
            axes[1].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=1)
            plt.tight_layout()
            plt.show()

        return sample_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Read Hyper-parameters
    with open('jsonfiles/supnerf.kitti.car.json', 'r') as f:
        hpams = json.load(f)

    kitti_data_dir = hpams['dataset']['data_dir']
    # det3d_path = 'data/third_party_det3D/fcos3d/result_kitti'

    kitti_dataset = KittiData(
        hpams,
        kitti_data_dir,
        split='val',
        debug=True,
        add_pose_err=3,
        init_rot_err=0.01,
        init_trans_err=0.3,
        # det3d_path=det3d_path,
    )

    dataloader = DataLoader(kitti_dataset, batch_size=1, num_workers=0, shuffle=True)

    occlusion_all = []
    distance_all = []
    # Analysis of valid portion of data
    for batch_idx, batch_data in enumerate(dataloader):
        obj_pose = batch_data['obj_poses']
        occlusion = batch_data['occlusion']
        dist = np.linalg.norm(obj_pose[0, :, 3].numpy())
        distance_all.append(dist)
        occlusion_all.append(occlusion[0].item())
        print(f'Finish {batch_idx} / {len(dataloader)}')

    # histogram of distance
    n, bins, patches = plt.hist(x=np.array(distance_all), bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.title('Histogram of object distance')
    plt.savefig('eval_summary/kitti_dist_hist.pdf')
    plt.close()

    # histogram of occlusion level
    n, bins, patches = plt.hist(x=np.array(occlusion_all), bins=[0, 1, 2, 3], color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Occlusion')
    plt.ylabel('Counts')
    plt.title('Histogram of occlusion level')
    plt.savefig('eval_summary/kitti_occ_hist.pdf')

    """
        Observed invalid scenarios expected to discard:
            night (failure of instance prediction cross-domain)
            truncation (currently not included)
            general instance prediction failure
            too far-away
            too heavy occluded (some fully occluded case's annotation may come from the projection of another time's annotations for static object)
    """
