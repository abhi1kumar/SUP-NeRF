import random
import numpy as np
import torch
import json
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from cityscapesscripts.helpers.labels import name2label, trainId2label
import torchvision.transforms as Transforms


from data.NuScenes import data_splits_nusc
from utils import (pts_in_box_3d, get_random_pose2, corners_of_box, view_points, render_box, roi_resize,
                   roi_process, preprocess_img_square, prepare_pixel_samples)


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


def get_associate_box_3d(objects, tgt_mask, tgt_cat, cam_intrinsic):
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

    for ii, cls_label in enumerate(objects['classes']):
        if cls_label != tgt_cat.rsplit('.')[-1]:
            continue
        corners_2d = view_points(np.array(objects['corners_3d'][ii]).T, view=cam_intrinsic, normalize=True)[:2, :]
        min_x2 = corners_2d[0].min()
        min_y2 = corners_2d[1].min()
        max_x2 = corners_2d[0].max()
        max_y2 = corners_2d[1].max()
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


class NuScenesData:
    def __init__(self, hpams,
                 nusc_data_dir,
                 nusc_seg_dir,
                 nusc_version,
                 split='train',
                 out_gt_depth=True,
                 add_pose_err=0,
                 init_rot_err=0.2,
                 init_trans_err=0.1,
                 rand_angle_lim=np.pi/9,
                 det3d_path=None,
                 test_size=5000,
                 pred_box2d=False,
                 box2d_rz_ratio=1.2,
                 debug=False,
                 num_subset=1,
                 id_subset=0,
                 aug_box2d=False,
                 render_sz=None,
                 prepare_batch_rays=False
                 ):
        """
            Provide camera input and label per annotation per instance for the target category
            Object 'Instance' here respects the definition from NuScene dataset.
            Each instance of object does not go beyond a single scene.
            Each instance contains multiple annotations from different timestamps
            Each annotation could be projected to multiple camera inputs at the same timestamp.
        """
        self.nusc_cat = hpams['dataset']['nusc_cat']
        self.seg_cat = hpams['dataset']['seg_cat']
        self.divisor = hpams['dataset']['divisor']
        self.box_iou_th = hpams['dataset']['box_iou_th']
        self.max_dist = hpams['dataset']['max_dist']
        self.min_lidar_cnt = hpams['dataset']['min_lidar_cnt']
        self.mask_pixels = hpams['dataset']['mask_pixels']
        self.img_h = hpams['dataset']['img_h']
        self.img_w = hpams['dataset']['img_w']
        self.out_gt_depth = out_gt_depth
        self.debug = debug
        self.split = split
        self.det3d_path = det3d_path
        self.pred_box2d = pred_box2d
        self.box2d_rz_ratio = box2d_rz_ratio
        self.aug_box2d = True
        self.rand_flip = Transforms.RandomHorizontalFlip(p=0.5)
        self.colorjitter = Transforms.ColorJitter(brightness=0.3, contrast=0.3)
        # self.flip_and_color_jitter = Transforms.Compose([
        #     Transforms.RandomHorizontalFlip(p=0.5),
        #     Transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0),
        # ])
        self.hpams = hpams
        self.aug_box2d = aug_box2d
        self.prepare_batch_rays = prepare_batch_rays
        self.render_sz = render_sz

        # Get the dataset ready
        self.nusc_data_dir = nusc_data_dir
        self.nusc_seg_dir = nusc_seg_dir
        self.seg_type = 'instance'
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_data_dir, verbose=True)
        instance_all = self.nusc.instance
        self.all_valid_samples = []  # (anntoken, cam) pairs
        self.anntokens_per_ins = {}  # dict for each instance's annotokens
        self.instoken_per_ann = {}  # dict for each annotation's instoken
        self.sample_attr = {}  # save attributes of valid samples for fast retrieval at run time

        # Prepare index file for the valid samples for later efficient batch preparation
        subset_index_file = 'data/NuScenes/nusc.' + nusc_version + '.' + split + '.' + self.nusc_cat + '.json'
        if os.path.exists(subset_index_file):
            nusc_subset = json.load(open(subset_index_file))
            if nusc_subset['box_iou_th'] != self.box_iou_th or nusc_subset['max_dist'] != self.max_dist or nusc_subset[
                'mask_pixels'] != self.mask_pixels or nusc_subset['min_lidar_cnt'] != self.min_lidar_cnt or nusc_subset[
                'seg_type'] != self.seg_type:
                print('Different dataset config found! Re-preprocess the dataset to prepare indices of valid samples...')
                self.preprocess_dataset(self.nusc_cat, split, instance_all, nusc_version, subset_index_file)
                nusc_subset = json.load(open(subset_index_file))
            else:
                self.all_valid_samples = nusc_subset['all_valid_samples']
                self.anntokens_per_ins = nusc_subset['anntokens_per_ins']
                self.instoken_per_ann = nusc_subset['instoken_per_ann']
                self.sample_attr = nusc_subset['sample_attr']
                print('Loaded existing index file for valid samples.')

            # use a subset for test
            if split != 'train' and len(self.all_valid_samples) > test_size:
                if 'rand_data_ids' not in nusc_subset.keys() or len(nusc_subset['rand_data_ids']) != test_size:
                    rand_data_ids = np.random.permutation(len(self.all_valid_samples))[:test_size]
                    nusc_subset['rand_data_ids'] = rand_data_ids.tolist()
                    json.dump(nusc_subset, open(subset_index_file, 'w'), indent=4)
                    print('updated selected indices for testing data')
                else:
                    rand_data_ids = nusc_subset['rand_data_ids']
                self.all_valid_samples = [self.all_valid_samples[ii] for ii in rand_data_ids]
        else:
            print('No existing index file found! Preprocess the dataset to prepare indices of valid samples...')
            self.preprocess_dataset(self.nusc_cat, split, instance_all, nusc_version, subset_index_file)

        print('Preparing camera data dictionary for fast retrival given image name')
        self.cam_data_dict = {}
        for sample in self.nusc.sample_data:
            if 'CAM' in sample['channel']:
                self.cam_data_dict[os.path.basename(sample['filename'])] = sample

        # divide the whole dataset to subsets to process in multiple threads
        set_size = len(self.all_valid_samples) // num_subset
        self.all_valid_samples = self.all_valid_samples[id_subset * set_size : (id_subset + 1) * set_size]
        self.lenids = len(self.all_valid_samples)
        print(f'{self.lenids} annotations in {self.nusc_cat} category are included in dataloader.')

        # for adding error to pose
        self.add_pose_err = add_pose_err
        self.init_rot_err = init_rot_err
        self.init_trans_err = init_trans_err
        self.rand_angle_lim = rand_angle_lim

    def preprocess_dataset(self, nusc_cat, split, instance_all, nusc_version, subset_file):
        """
            Go through the full dataset once to save the valid indices. Save the index file for later direct refer.
        """

        # retrieve all the target instance
        for instance in tqdm(instance_all):
            if self.nusc.get('category', instance['category_token'])['name'] == nusc_cat:
                instoken = instance['token']
                # self.tgt_instance_list.append(instance)
                anntokens = self.nusc.field2token('sample_annotation', 'instance_token', instoken)
                for anntoken in anntokens:
                    # rule out those night samples
                    sample_ann = self.nusc.get('sample_annotation', anntoken)
                    sample_record = self.nusc.get('sample', sample_ann['sample_token'])
                    scene = self.nusc.get('scene', sample_record['scene_token'])
                    if 'mini' in nusc_version:
                        if split == 'train' and scene['name'] not in data_splits_nusc.mini_train:
                            continue
                        if split == 'val' and scene['name'] not in data_splits_nusc.mini_val:
                            continue
                    if 'trainval' in nusc_version:
                        if split == 'train' and scene['name'] not in data_splits_nusc.train:
                            continue
                        if split == 'val' and scene['name'] not in data_splits_nusc.val:
                            continue
                    if 'test' in nusc_version:
                        if split == 'test' and scene['name'] not in data_splits_nusc.test:
                            continue

                    log_file = self.nusc.get('log', scene['log_token'])['logfile']
                    log_items = log_file.split('-')
                    if int(log_items[4]) >= 18:  # Consider time after 18:00 as night
                        continue

                    # check those qualified samples
                    if 'LIDAR_TOP' in sample_record['data'].keys():
                        cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
                        cams = np.random.permutation(cams)
                        for cam in cams:
                            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                                           box_vis_level=BoxVisibility.ALL,
                                                                                           selected_anntokens=[anntoken])
                            if len(boxes) == 1:
                                # box here is in sensor coordinate system
                                box = boxes[0]
                                obj_center = box.center
                                corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                                min_x = np.min(corners[0, :])
                                max_x = np.max(corners[0, :])
                                min_y = np.min(corners[1, :])
                                max_y = np.max(corners[1, :])
                                box_2d = [min_x, min_y, max_x, max_y]

                                # Prepare gt depth map using lidar points
                                pointsensor_token = sample_record['data']['LIDAR_TOP']
                                camtoken = sample_record['data'][cam]

                                # Only the lider points within the target camera image belong to the target class is returned.
                                lidar_pts_im, lider_pts_depth, _ = self.nusc.explorer.map_pointcloud_to_image(
                                    pointsensor_token, camtoken,
                                    render_intensity=False,
                                    show_lidarseg=False,
                                    filter_lidarseg_labels=None,
                                    lidarseg_preds_bin_path=None,
                                    show_panoptic=False)

                                lidar_pts_cam = np.matmul(np.linalg.inv(camera_intrinsic),
                                                          lidar_pts_im) * lider_pts_depth
                                pts_ann_indices = pts_in_box_3d(lidar_pts_cam, box.corners(), keep_top_portion=0.9)
                                lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]

                                json_file = os.path.join(self.nusc_seg_dir, cam,
                                                            os.path.basename(data_path)[:-4] + '.json')
                                preds = json.load(open(json_file))
                                ins_masks = []
                                for box_id in range(0, len(preds['boxes'])):
                                    mask_file = os.path.join(self.nusc_seg_dir, cam,
                                                                os.path.basename(data_path)[:-4] + f'_{box_id}.png')
                                    mask = np.asarray(Image.open(mask_file))
                                    ins_masks.append(mask)

                                tgt_ins_id, tgt_ins_cnt, area_ratio, box_iou, lidar_cnt = get_tgt_ins_from_masksrcnn_v2(
                                    preds,
                                    ins_masks,
                                    self.seg_cat,
                                    box_2d,
                                    lidar_pts_im_ann)

                                # save the qualified sample index for later direct use
                                if tgt_ins_id is not None and tgt_ins_cnt > self.mask_pixels and box_iou > self.box_iou_th and area_ratio > self.box_iou_th and np.linalg.norm(
                                        obj_center) < self.max_dist and lidar_cnt >= self.min_lidar_cnt:
                                    self.all_valid_samples.append([anntoken, cam])
                                    if instoken not in self.anntokens_per_ins.keys():
                                        self.anntokens_per_ins[instoken] = [[anntoken, cam]]
                                    else:
                                        self.anntokens_per_ins[instoken].append([anntoken, cam])
                                    if anntoken not in self.instoken_per_ann.keys():
                                        self.instoken_per_ann[anntoken] = instoken
                                    if anntoken not in self.sample_attr.keys():
                                        self.sample_attr[anntoken] = {}
                                    if cam not in self.sample_attr[anntoken].keys():
                                        self.sample_attr[anntoken][cam] = {'seg_id': tgt_ins_id, 'lidar_cnt': lidar_cnt}

        # save into json file for quick load next time
        nusc_subset = {}
        nusc_subset['all_valid_samples'] = self.all_valid_samples
        nusc_subset['anntokens_per_ins'] = self.anntokens_per_ins
        nusc_subset['instoken_per_ann'] = self.instoken_per_ann
        nusc_subset['sample_attr'] = self.sample_attr
        nusc_subset['box_iou_th'] = self.box_iou_th
        nusc_subset['max_dist'] = self.max_dist
        nusc_subset['mask_pixels'] = self.mask_pixels
        nusc_subset['min_lidar_cnt'] = self.min_lidar_cnt
        nusc_subset['seg_type'] = self.seg_type

        json.dump(nusc_subset, open(subset_file, 'w'), indent=4)

    def __len__(self):
        return self.lenids

    def __getitem__(self, idx):
        sample_data = {}
        anntoken, cam = self.all_valid_samples[idx]
        if self.debug:
            print(f'anntoken: {anntoken}')

        # For each annotation (one annotation per timestamp) get all the sensors
        sample_ann = self.nusc.get('sample_annotation', anntoken)
        sample_record = self.nusc.get('sample', sample_ann['sample_token'])

        # Figure out which camera the object is fully visible in (this may return nothing).
        if self.debug:
            print(f'     {cam}')
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                       box_vis_level=BoxVisibility.ALL,
                                                                       selected_anntokens=[anntoken])
        # Plot CAMERA view.
        img = Image.open(data_path)
        img = np.asarray(img)

        # box here is in sensor coordinate system
        box = boxes[0]
        # compute the camera pose in object frame, make sure dataset and model definitions consistent
        obj_center = box.center
        obj_orientation = box.orientation.rotation_matrix
        obj_pose = np.concatenate([obj_orientation, np.expand_dims(obj_center, -1)], axis=1)
        # Compute camera pose in object frame = c2o transformation matrix
        # Recall that object pose in camera frame = o2c transformation matrix
        R_c2o = obj_orientation.transpose()
        t_c2o = - R_c2o @ np.expand_dims(obj_center, -1)
        cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)

        # find the valid instance given 2d box projection
        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
        min_x = np.min(corners[0, :])
        max_x = np.max(corners[0, :])
        min_y = np.min(corners[1, :])
        max_y = np.max(corners[1, :])
        box_2d = [min_x, min_y, max_x, max_y]


        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        for box_id in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + f'_{box_id}.png')
            mask = np.asarray(Image.open(mask_file))
            ins_masks.append(mask)
        if len(ins_masks) == 0:
            mask_occ = None
        else:
            tgt_ins_id = self.sample_attr[anntoken][cam]['seg_id']
            mask_occ = get_mask_occ_from_ins(ins_masks, tgt_ins_id)
            if self.pred_box2d:
                box_2d = preds['boxes'][tgt_ins_id]
                # enlarge pred_box
                box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)
        lidar_cnt = self.sample_attr[anntoken][cam]['lidar_cnt']

        # TODO: currently not synced with box_2d, so the crop image might not be centered perfectly
        if self.add_pose_err == 1:
            # only consider yaw error and distance error
            # yaw_err = random.uniform(-self.max_rot_pert, self.max_rot_pert)
            yaw_err = random.choice([1., -1.]) * self.init_rot_err
            rot_err = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                                [np.sin(yaw_err), np.cos(yaw_err), 0.],
                                [0., 0., 1.]]).astype(np.float32)
            # trans_err_ratio = random.uniform(1.0-self.max_t_pert, 1.0+self.max_t_pert)
            trans_err_ratio = 1. + random.choice([1., -1.]) * self.init_trans_err
            obj_center_w_err = obj_center * trans_err_ratio
            obj_orientation_w_err = obj_orientation @ rot_err  # rot error need to right --> to model points
            obj_pose_w_err = np.concatenate([obj_orientation_w_err, np.expand_dims(obj_center_w_err, -1)], axis=1)
            R_c2o_w_err = obj_orientation_w_err.transpose()
            t_c2o_w_err = -R_c2o_w_err @ np.expand_dims(obj_center_w_err, -1)
            cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)

            sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose_w_err.astype(np.float32))
            sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose_w_err.astype(np.float32))
        elif self.add_pose_err >= 2:
            obj_pose_w_err = get_random_pose2(camera_intrinsic.astype(np.float32),
                                              np.asarray(box_2d).astype(np.int32),
                                              yaw_lim=np.pi, angle_lim=self.rand_angle_lim,
                                              trans_lim=0.3, depth_fix=20)
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
            det_file = os.path.join(self.det3d_path, cam, os.path.basename(data_path)[:-4] + '.json')
            objects_pred = json.load(open(det_file))

            asso_obx_id, box_iou = get_associate_box_3d(objects_pred,
                                                        ins_masks[tgt_ins_id],
                                                        self.nusc_cat,
                                                        camera_intrinsic)
            if asso_obx_id >= 0 and box_iou > 0:
                ry_pred = objects_pred['boxes_yaw'][asso_obx_id]
                R_yaw = np.array([[np.cos(ry_pred), 0., np.sin(ry_pred)],
                                  [0., 1., 0.],
                                  [-np.sin(ry_pred), 0., np.cos(ry_pred)]]).astype(np.float32)
                # R_obj = R_yaw
                R_unit = np.array([[1., 0., 0.],
                                   [0., 0., -1.],
                                   [0., 1., 0.]])
                R_obj = R_yaw @ R_unit  # TODO: this is wield because such R_unit is not same as nusc box
                T_obj = np.asarray(objects_pred['boxes_center'][asso_obx_id]).reshape(3, 1)
                # wlh_pred = np.asarray(objects_pred['boxes_dims'][asso_obx_id])
                # wlh_pred = wlh_pred[[2, 0, 1]]
                # T_obj[1] -= wlh_pred[2]/2
                # wlh = self.nusc.get('sample_annotation', anntoken)['size']
                # T_obj[1] -= wlh[2]/2
                obj_pose_w_err = np.concatenate([R_obj, T_obj], axis=1)
                # Compute camera pose in object frame = c2o transformation matrix
                R_c2o = R_obj.transpose()
                t_c2o = - R_c2o @ T_obj
                cam_pose_w_err = np.concatenate([R_c2o, t_c2o], axis=1)
                sample_data['cam_poses_w_err'] = torch.from_numpy(cam_pose_w_err.astype(np.float32))
                sample_data['obj_poses_w_err'] = torch.from_numpy(obj_pose_w_err.astype(np.float32))


        # Prepare gt depth map using lidar points
        pointsensor_token = sample_record['data']['LIDAR_TOP']
        camtoken = sample_record['data'][cam]
        # lidarseg_idx = self.nusc.lidarseg_name2idx_mapping[self.nusc_cat]

        if self.out_gt_depth or self.debug:
            # Only the lider points within the target camera image belong to the target class is returned.
            lidar_pts_im, lider_pts_depth, _ = self.nusc.explorer.map_pointcloud_to_image(pointsensor_token, camtoken,
                                                                                          render_intensity=False,
                                                                                          show_lidarseg=False,
                                                                                          filter_lidarseg_labels=None,
                                                                                          lidarseg_preds_bin_path=None,
                                                                                          show_panoptic=False)

            lidar_pts_cam = np.matmul(np.linalg.inv(camera_intrinsic), lidar_pts_im) * lider_pts_depth
            pts_ann_indices = pts_in_box_3d(lidar_pts_cam, box.corners(), keep_top_portion=0.9)
            lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]
            lider_pts_depth_ann = lider_pts_depth[pts_ann_indices]

            depth_map = np.zeros(img.shape[:2]).astype(np.float32)
            depth_map[lidar_pts_im_ann[1, :].astype(np.int32), lidar_pts_im_ann[0, :].astype(np.int32)] = lider_pts_depth_ann
            sample_data['depth_maps'] = torch.from_numpy(depth_map.astype(np.float32))

        # ATTENTION: prepare batch data including ray based samples can further improve efficiency,
        # but lower flexible for training considering different crop sizes

        sample_data['imgs'] = torch.from_numpy(img.astype(np.float32)/255.)
        sample_data['masks_occ'] = torch.from_numpy(mask_occ.astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(box_2d).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(camera_intrinsic.astype(np.float32))
        sample_data['cam_poses'] = torch.from_numpy(np.asarray(cam_pose).astype(np.float32))
        sample_data['obj_poses'] = torch.from_numpy(np.asarray(obj_pose).astype(np.float32))
        sample_data['instoken'] = self.instoken_per_ann[anntoken]
        sample_data['anntoken'] = anntoken
        sample_data['cam_ids'] = cam
        wlh = self.nusc.get('sample_annotation', anntoken)['size']
        sample_data['wlh'] = torch.tensor(wlh, dtype=torch.float32)

        if self.prepare_batch_rays:
            img = sample_data['imgs'].clone()
            mask_occ = sample_data['masks_occ'].clone()
            roi = sample_data['rois'].clone()
            obj_diag = np.linalg.norm(wlh).astype(np.float32)
            if self.aug_box2d:
                rz_ratio = random.uniform(0.9, 1.1)
                roi = roi_resize(roi, ratio=rz_ratio)
                roi = torch.stack(roi)
                rand_move = random.uniform(-5, 5)
                roi += rand_move
                roi = roi.type(torch.int)

            # got square ROI within image area
            if self.render_sz is not None:
                roi = roi_process(roi, self.img_h, self.img_w, self.hpams['roi_margin'], sq_pad=True)
            else:
                roi = roi_process(roi, self.img_h, self.img_w, self.hpams['roi_margin'], sq_pad=False)
            # crop tgt img to roi
            img = img[roi[1]: roi[3], roi[0]: roi[2]]
            mask_occ = mask_occ[roi[1]: roi[3], roi[0]: roi[2]].unsqueeze(-1)
            # only keep the fg portion, but turn BG to white (for ShapeNet Pretrained model)
            img = img * (mask_occ > 0)
            # if self.hpams['white_bkgd']:
            #     img = img + (mask_occ <= 0)

            # Preprocess img for model inference (pad and resize to the same square size)
            img_in = preprocess_img_square(img, self.hpams['in_img_sz'])
            xyz, viewdir, z_vals, rgb_tgt, occ_pixels = prepare_pixel_samples(img, mask_occ, sample_data['cam_poses'],
                                                                              obj_diag, sample_data['cam_intrinsics'],
                                                                              roi, self.hpams['n_rays'],
                                                                              self.hpams['n_samples'],
                                                                              self.hpams['shapenet_obj_cood'],
                                                                              self.hpams['sym_aug'],
                                                                              im_sz=self.render_sz)

            sample_data['img_in'] = img_in.squeeze()
            sample_data['xyz'] = xyz
            sample_data['viewdir'] = viewdir
            sample_data['z_vals'] = z_vals
            sample_data['rgb_tgt'] = rgb_tgt
            sample_data['occ_pixels'] = occ_pixels
            # update changed rois
            sample_data['rois'] = roi.int()

        if self.debug:
            print(
                f'        tgt instance id: {tgt_ins_id}, '
                f'lidar pts cnt: {lidar_cnt} ')

            camtoken = sample_record['data'][cam]
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))

            # draw object box on the image
            img2 = np.copy(img)
            corners_3d = corners_of_box(obj_pose, wlh, is_kitti=False)
            pred_uv = view_points(
                corners_3d,
                camera_intrinsic, normalize=True)
            c = np.array([0, 255, 0]).astype(np.float)
            img2 = render_box(img2, pred_uv, colors=(c, c, c))
            if self.add_pose_err > 0:
                corners_3d_w_err = corners_of_box(obj_pose_w_err, wlh, is_kitti=False)
                # corners_3d_w_err = np.array(objects_pred['corners_3d'][asso_obx_id]).T
                pred_uv_w_err = view_points(
                    corners_3d_w_err,
                    camera_intrinsic, normalize=True)
                c = np.array([255, 0, 0]).astype(np.float)
                img2 = render_box(img2, pred_uv_w_err, colors=(c, c, c))
            axes[0].imshow(img2)
            axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
            axes[0].axis('off')
            axes[0].set_aspect('equal')
            # c = np.array(self.nusc.colormap[box.name]) / 255.0
            # box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            seg_vis = ins2vis(ins_masks, tgt_ins_id)
            axes[1].imshow(seg_vis)
            axes[1].set_title('pred instance')
            axes[1].axis('off')
            axes[1].set_aspect('equal')
            # c = np.array(nusc.colormap[box.name]) / 255.0
            min_x, min_y, max_x, max_y = box_2d
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                     linewidth=2, edgecolor='y', facecolor='none')
            axes[1].add_patch(rect)

            axes[0].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)
            axes[1].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)

            self.nusc.render_annotation(anntoken, margin=30, box_vis_level=BoxVisibility.ALL)
            plt.tight_layout()
            plt.show()
            print(f"        Lidar pts in target segment: {lidar_cnt}")
            # Nusc claimed pixel ratio visible from 6 cameras, seem not very reliable since no GT amodel segmentation
            visibility_token = sample_ann['visibility_token']
            print("        Visibility: {}".format(self.nusc.get('visibility', visibility_token)))

        return sample_data

    # TODO: Prepare gt depth map from lidar not included yet
    def get_ins_samples(self, instoken):
        samples = {}
        samples_in = self.anntokens_per_ins[instoken]
        # extract fixed number of qualified samples per instance
        imgs = []
        masks_occ = []
        depth_maps = []
        cam_poses = []
        cam_poses_w_err = []
        obj_poses = []
        obj_poses_w_err = []
        cam_intrinsics = []
        rois = []  # used to sample rays
        out_anntokens = []
        cam_ids = []
        wlh_list = []

        for sample_in in samples_in:
            [anntoken, cam] = sample_in
            if self.debug:
                print(f'instance: {instoken}, anntoken: {anntoken}')

            # For each annotation (one annotation per timestamp) get all the sensors
            sample_ann = self.nusc.get('sample_annotation', anntoken)
            sample_record = self.nusc.get('sample', sample_ann['sample_token'])

            if self.debug:
                print(f'     {cam}')
            # TODO: consider BoxVisibility.ANY?
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                           box_vis_level=BoxVisibility.ALL,
                                                                           selected_anntokens=[anntoken])
            if len(boxes) == 1:
                # Plot CAMERA view.
                img = Image.open(data_path)
                img = np.asarray(img)

                # box here is in sensor coordinate system
                box = boxes[0]
                # compute the camera pose in object frame, make sure dataset and model definitions consistent
                obj_center = box.center
                obj_orientation = box.orientation.rotation_matrix
                obj_pose = np.concatenate([obj_orientation, np.expand_dims(obj_center, -1)], axis=1)

                # Compute camera pose in object frame = c2o transformation matrix
                # Recall that object pose in camera frame = o2c transformation matrix
                R_c2o = obj_orientation.transpose()
                t_c2o = - R_c2o @ np.expand_dims(obj_center, -1)
                cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)
                # find the valid instance given 2d box projection
                corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                min_x = np.min(corners[0, :])
                max_x = np.max(corners[0, :])
                min_y = np.min(corners[1, :])
                max_y = np.max(corners[1, :])
                box_2d = [min_x, min_y, max_x, max_y]

                # get mask
                json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.json')
                preds = json.load(open(json_file))
                ins_masks = []
                for box_id in range(0, len(preds['boxes'])):
                    mask_file = os.path.join(self.nusc_seg_dir, cam,
                                                os.path.basename(data_path)[:-4] + f'_{box_id}.png')
                    mask = np.asarray(Image.open(mask_file))
                    ins_masks.append(mask)

                if len(ins_masks) == 0:
                    mask_occ = None
                else:
                    tgt_ins_id = self.sample_attr[anntoken][cam]['seg_id']
                    mask_occ = get_mask_occ_from_ins(ins_masks, tgt_ins_id)
                    if self.pred_box2d:
                        box_2d = preds['boxes'][tgt_ins_id]
                        # enlarge pred_box
                        box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)
                lidar_cnt = self.sample_attr[anntoken][cam]['lidar_cnt']


                # ATTENTION: add Rot error in the object's coordinate, and T error
                if self.add_pose_err == 1:
                    # only consider yaw error and distance error
                    # yaw_err = random.uniform(-self.max_rot_pert, self.max_rot_pert)
                    yaw_err = random.choice([1., -1.]) * self.init_rot_err
                    rot_err = np.array([[np.cos(yaw_err), -np.sin(yaw_err), 0.],
                                        [np.sin(yaw_err), np.cos(yaw_err), 0.],
                                        [0., 0., 1.]]).astype(np.float32)
                    # trans_err_ratio = random.uniform(1.0-self.max_t_pert, 1.0+self.max_t_pert)
                    trans_err_ratio = 1. + random.choice([1., -1.]) * self.init_trans_err
                    obj_center_w_err = obj_center * trans_err_ratio
                    obj_orientation_w_err = obj_orientation @ rot_err# rot error need to right --> to model points
                    obj_pose_w_err = np.concatenate([obj_orientation_w_err, np.expand_dims(obj_center_w_err, -1)], axis=1)
                    R_c2o_w_err = obj_orientation_w_err.transpose()
                    # ATTENTION: t_c2o_w_err is proportional to t_c2o because added error to R
                    t_c2o_w_err = -R_c2o_w_err @ np.expand_dims(obj_center_w_err, -1)
                    cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)
                    # TODO: not synced with box_2d
                elif self.add_pose_err >= 2:
                    # obj_pose_w_err = get_random_pose(obj_pose,
                    #                                  camera_intrinsic.astype(np.float32),
                    #                                  np.asarray(box_2d).astype(np.int32),
                    #                                  yaw_lim=np.pi / 2, angle_lim=np.pi / 9,
                    #                                  trans_lim=0.3, depth_lim=0.3)
                    obj_pose_w_err = get_random_pose2(camera_intrinsic.astype(np.float32),
                                                      np.asarray(box_2d).astype(np.int32),
                                                      yaw_lim=np.pi, angle_lim=self.rand_angle_lim,
                                                      trans_lim=0.3, depth_fix=20)
                    R_c2o_w_err = obj_pose_w_err[:3, :3].T
                    t_c2o_w_err = -R_c2o_w_err @ obj_pose_w_err[:3, 3:]
                    cam_pose_w_err = np.concatenate([R_c2o_w_err, t_c2o_w_err], axis=1)

                # Associate with third-party detection results (Separate condition in case no association exists)
                if self.add_pose_err == 3:
                    det_file = os.path.join(self.det3d_path, cam, os.path.basename(data_path)[:-4] + '.json')
                    objects_pred = json.load(open(det_file))

                    asso_obx_id, box_iou = get_associate_box_3d(objects_pred,
                                                                ins_masks[tgt_ins_id],
                                                                self.nusc_cat,
                                                                camera_intrinsic)
                    if asso_obx_id >= 0 and box_iou > 0:
                        ry_pred = objects_pred['boxes_yaw'][asso_obx_id]
                        R_yaw = np.array([[np.cos(ry_pred), 0., np.sin(ry_pred)],
                                          [0., 1., 0.],
                                          [-np.sin(ry_pred), 0., np.cos(ry_pred)]]).astype(np.float32)
                        # R_obj = R_yaw
                        R_unit = np.array([[1., 0., 0.],
                                           [0., 0., -1.],
                                           [0., 1., 0.]])
                        R_obj = R_yaw @ R_unit  # TODO: this is wield because such R_unit is not same as nusc box
                        T_obj = np.asarray(objects_pred['boxes_center'][asso_obx_id]).reshape(3, 1)
                        # wlh_pred = np.asarray(objects_pred['boxes_dims'][asso_obx_id])
                        # wlh_pred = wlh_pred[[2, 0, 1]]
                        # T_obj[1] -= wlh_pred[2]/2
                        # wlh = self.nusc.get('sample_annotation', anntoken)['size']
                        # T_obj[1] -= wlh[2]/2
                        obj_pose_w_err = np.concatenate([R_obj, T_obj], axis=1)
                        # Compute camera pose in object frame = c2o transformation matrix
                        R_c2o = R_obj.transpose()
                        t_c2o = - R_c2o @ T_obj
                        cam_pose_w_err = np.concatenate([R_c2o, t_c2o], axis=1)

                # Prepare gt depth map using lidar points
                pointsensor_token = sample_record['data']['LIDAR_TOP']
                camtoken = sample_record['data'][cam]
                # lidarseg_idx = self.nusc.lidarseg_name2idx_mapping[self.nusc_cat]

                if self.out_gt_depth or self.debug:
                    # Only the lider points within the target camera image belong to the target class is returned.
                    lidar_pts_im, lider_pts_depth, _ = self.nusc.explorer.map_pointcloud_to_image(pointsensor_token,
                                                                                                  camtoken,
                                                                                                  render_intensity=False,
                                                                                                  show_lidarseg=False,
                                                                                                  filter_lidarseg_labels=None,
                                                                                                  lidarseg_preds_bin_path=None,
                                                                                                  show_panoptic=False)

                    lidar_pts_cam = np.matmul(np.linalg.inv(camera_intrinsic), lidar_pts_im) * lider_pts_depth
                    pts_ann_indices = pts_in_box_3d(lidar_pts_cam, box.corners(), keep_top_portion=0.9)
                    lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]
                    lider_pts_depth_ann = lider_pts_depth[pts_ann_indices]

                    depth_map = np.zeros(img.shape[:2]).astype(np.float32)
                    depth_map[lidar_pts_im_ann[1, :].astype(np.int32), lidar_pts_im_ann[0, :].astype(
                        np.int32)] = lider_pts_depth_ann
                    depth_maps.append(depth_map.astype(np.float32))



                imgs.append(img)
                masks_occ.append(mask_occ.astype(np.int32))
                # masks.append((pan_label == tgt_ins_id).astype(np.int32))
                rois.append(box_2d)
                cam_intrinsics.append(camera_intrinsic)
                cam_poses.append(cam_pose)
                obj_poses.append(obj_pose)
                out_anntokens.append(anntoken)
                cam_ids.append(cam)
                wlh = self.nusc.get('sample_annotation', anntoken)['size']
                wlh_list.append(wlh)

                if self.add_pose_err:
                    cam_poses_w_err.append(cam_pose_w_err)
                    obj_poses_w_err.append(obj_pose_w_err)

                if self.debug:
                    print(
                        f'        tgt instance id: {tgt_ins_id}, '
                        f'lidar pts cnt: {lidar_cnt} ')

                    camtoken = sample_record['data'][cam]
                    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
                    axes[0].imshow(img)
                    axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
                    axes[0].axis('off')
                    axes[0].set_aspect('equal')
                    c = np.array(self.nusc.colormap[box.name]) / 255.0
                    box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    seg_vis = ins2vis(ins_masks, tgt_ins_id)
                    axes[1].imshow(seg_vis)
                    axes[1].set_title('pred instance')
                    axes[1].axis('off')
                    axes[1].set_aspect('equal')
                    # c = np.array(nusc.colormap[box.name]) / 255.0
                    min_x, min_y, max_x, max_y = box_2d
                    rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                             linewidth=2, edgecolor='y', facecolor='none')
                    axes[1].add_patch(rect)

                    axes[0].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)
                    axes[1].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)
                    plt.tight_layout()
                    plt.show()

                    self.nusc.render_annotation(anntoken, margin=30, box_vis_level=BoxVisibility.ALL)
                    print(f"        Lidar pts in target segment: {lidar_cnt}")
                    # Nusc claimed pixel ratio visible from 6 cameras, seem not very reliable since no GT amodel segmentation
                    visibility_token = sample_ann['visibility_token']
                    print("        Visibility: {}".format(self.nusc.get('visibility', visibility_token)))

        samples['imgs'] = torch.from_numpy(np.asarray(imgs).astype(np.float32) / 255.)
        samples['masks_occ'] = torch.from_numpy(np.asarray(masks_occ).astype(np.float32))
        samples['rois'] = torch.from_numpy(np.asarray(rois).astype(np.int32))
        samples['cam_intrinsics'] = torch.from_numpy(np.asarray(cam_intrinsics).astype(np.float32))
        samples['cam_poses'] = torch.from_numpy(np.asarray(cam_poses).astype(np.float32))
        samples['obj_poses'] = torch.from_numpy(np.asarray(obj_poses).astype(np.float32))
        samples['anntokens'] = out_anntokens
        samples['cam_ids'] = cam_ids
        samples['wlh'] = torch.from_numpy(np.asarray(wlh_list).astype(np.float32))

        if self.out_gt_depth:
            samples['depth_maps'] = torch.from_numpy(np.asarray(depth_maps).astype(np.float32))

        if self.add_pose_err:
            samples['cam_poses_w_err'] = torch.from_numpy(np.asarray(cam_poses_w_err).astype(np.float32))
            samples['obj_poses_w_err'] = torch.from_numpy(np.asarray(obj_poses_w_err).astype(np.float32))

        return samples

    def get_objects_in_image(self, filename):
        """
            Output objects annotations pere image
        """
        if filename not in self.cam_data_dict.keys():
            print(f'Target image file {filename} does not contain valid annotations')
            return None

        cam_data = self.cam_data_dict[filename]
        # load image, 2D boxes and masks, only need to get K from nusc
        impath, _, camera_intrinsic = self.nusc.get_sample_data(cam_data['token'], box_vis_level=BoxVisibility.ANY)

        # load image
        img = Image.open(impath)
        img = np.asarray(img)

        # load mask-rcnn predicted instance masks and 2D boxes
        cam = cam_data['channel']
        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        rois = []
        for ii in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + f'_{ii}.png')
            mask = np.asarray(Image.open(mask_file))
            if self.seg_cat in preds['labels'][ii]:
                ins_masks.append(mask)
                box_2d = preds['boxes'][ii]
                # enlarge pred_box
                box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)
                rois.append(box_2d)
        if len(rois) == 0:
            print('No valid objects found in the Image!')
            return None

        masks_occ = []
        for ii in range(0, len(ins_masks)):
            mask_occ = get_mask_occ_from_ins(ins_masks, ii)
            masks_occ.append(mask_occ)

        # Need to predict whl from trained model
        if self.debug:
            self.nusc.render_sample_data(cam_data['token'])
            plt.show()

        # output data
        sample_data = {}
        sample_data['img'] = torch.from_numpy(img.astype(np.float32) / 255.)
        sample_data['masks_occ'] = torch.from_numpy(np.asarray(masks_occ).astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(rois).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(camera_intrinsic.astype(np.float32))
        return sample_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Read Hyper-parameters
    with open('jsonfiles/supnerf.nusc.vehicle.car.json', 'r') as f:
        hpams = json.load(f)

    nusc_data_dir = hpams['dataset']['test_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_instance')
    nusc_version = hpams['dataset']['test_nusc_version']
    det3d_path = os.path.join(nusc_data_dir, 'pred_det3d')

    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='val',
        debug=True,
        add_pose_err=0,
        det3d_path=det3d_path,
        pred_box2d=False,
        selfsup=False
    )

    dataloader = DataLoader(nusc_dataset, batch_size=1, num_workers=0, shuffle=True)

    # get predicted objects and masks associated with each image
    objects_data = nusc_dataset.get_objects_in_image('n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984240912467.jpg')

    # Analysis of valid portion of data
    distance_all = []
    visibility_all = []
    wlh_all = []
    for batch_idx, batch_data in enumerate(dataloader):
        instoken = batch_data['instoken']
        anntoken = batch_data['anntoken']
        obj_pose = batch_data['obj_poses']
        wlh_batch = batch_data['wlh']

        sample_ann = nusc_dataset.nusc.get('sample_annotation', anntoken[0])
        visibility_token = sample_ann['visibility_token']
        visibility = nusc_dataset.nusc.get('visibility', visibility_token)

        dist = np.linalg.norm(obj_pose[0, :, 3].numpy())
        distance_all.append(dist)
        visibility_all.append(int(visibility['token']))
        wlh_all.append((wlh_batch[0].numpy()))
        print(f'Finish {batch_idx} / {len(dataloader)}')

    # compute the stats of wlh
    wlh_all = np.array(wlh_all)
    wlh_mean = np.mean(wlh_all, axis=0)
    wlh_std = np.std(wlh_all, axis=0)
    print(f'wlh mean: {wlh_mean},  wlh std: {wlh_std}')

    # histogram of distance
    n, bins, patches = plt.hist(x=np.array(distance_all), bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.title('Histogram of object distance')
    plt.savefig('eval_summary/nusc_dist_hist.pdf')
    plt.close()

    # histogram of occlusion level
    n, bins, patches = plt.hist(x=visibility_all, bins=[1, 2, 3, 4, 5], color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Visibility (6 CAM)')
    plt.ylabel('Counts')
    plt.title('Histogram of visibility level')
    plt.savefig('eval_summary/nusc_vis_hist.pdf')


    """
        Observed invalid scenarios expected to discard:
            night (failure of instance prediction cross-domain)
            truncation (currently not included)
            general instance prediction failure
            too far-away
            too heavy occluded (some fully occluded case's annotation may come from the projection of another time's annotations for static object)
    """
