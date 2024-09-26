import sys, os
ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from src.utils import str2bool
from src.optimizer_nuscenes import OptimizerNuScenes
from src.data_nuscenes import NuScenesData
from src.utils import collect_eval_results


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    # arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/autorfmix.nusc.vehicle.car.json")
    arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/supnerf.nusc.vehicle.car.json")
    arg_parser.add_argument("--model_epoch", dest="model_epoch", default=39, help="Specify certain epoch model to load")
    arg_parser.add_argument("--seg_source", dest="seg_source", default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--opt_multiview", dest="opt_multiview", default=False,
                            help="if to optimize multiple annotations of the same instance jointly")
    arg_parser.add_argument("--opt_pose", dest="opt_pose", type=int, default=1,
                            help="0: given init pose, optimize codes only."
                                 "1: given init pose, optimize both pose and codes. ")
    arg_parser.add_argument("--reg_iters", dest="reg_iters", type=int, default=3,
                            help="number of iters to use dl regressor to update."
                                 "If 0, pose updates purely rely on Nerf BP")
    arg_parser.add_argument("--add_pose_err", dest="add_pose_err", type=int, default=2,
                            help="0: no error pose from dataset"
                                 "1: controlled pose err from dataset"
                                 "2: full range pose err in reasonable range"
                                 "3: use associated third-party 3d detection results with target mask")
    arg_parser.add_argument("--init_rot_err", dest="init_rot_err", type=float, default=0.0,
                            help="Apply initial error of rotation in radians")
    arg_parser.add_argument("--init_trans_err", dest="init_trans_err", type=float, default=0.2,
                            help="Apply initial error of translation in ratio of distance to object center")
    arg_parser.add_argument("--rand_angle_lim", dest="rand_angle_lim", type=float, default=0,
                            help="normally 0 for test case, np.pi/9 is normally used for training")
    arg_parser.add_argument("--pred_box2d", dest="pred_box2d", type=int, default=0,
                            help="whether to use predicted 2d box from maskrcnn for optimization")
    arg_parser.add_argument("--pred_wlh", dest="pred_wlh", type=int, default=0,
                            help="whether to use wlh of 3d box from nerf model")
    arg_parser.add_argument("--vis", dest="vis", type=int, default=1,
                            help="0: no image saved; 1: save start and end frame; 2: save all frames")
    arg_parser.add_argument("--cross_eval_folder", dest="cross_eval_folder",
                            default=None,
                            help="the previously saved folder to conduct cross-view evaluation. If given skip optimize")
    arg_parser.add_argument("--num_subset", dest="num_subset", type=int, default=1,
                            help="number of subsets the whole dataset is divided into.")
    arg_parser.add_argument("--id_subset", dest="id_subset", type=int, default=0,
                            help="the id of subset to process.")
    arg_parser.add_argument("--nusc-version", dest="nusc_version", type=str, default='v1.0-trainval')
    args = arg_parser.parse_args()

    # Read Hyper-parameters
    with open(args.config_file, 'r') as f:
        hpams = json.load(f)

    hpams_pose_refiner = None
    hpams_pose_regressor = None

    nusc_data_dir = hpams['dataset']['test_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_' + args.seg_source)
    # nusc_version = hpams['dataset']['test_nusc_version']
    det3d_path = os.path.join(nusc_data_dir, 'pred_det3d')

    # create dataset
    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        args.nusc_version,
        split='val',
        debug=False,
        add_pose_err=args.add_pose_err,
        init_rot_err=args.init_rot_err,
        init_trans_err=args.init_trans_err,
        rand_angle_lim=args.rand_angle_lim,
        det3d_path=det3d_path,
        pred_box2d=args.pred_box2d,
        num_subset=args.num_subset,
        id_subset=args.id_subset
    )

    # create optimizer
    save_postfix = '_nuscenes'
    if args.opt_multiview:
        save_postfix += '_multiview'
        code_level = 0  # save at instance level, cross view, not cross scene
    else:
        code_level = 2

    save_postfix += f'_opt_pose_{args.opt_pose}'

    if args.add_pose_err == 1:
        save_postfix += f'_rot_err_{args.init_rot_err}_trans_err_{args.init_trans_err}'
    elif args.add_pose_err == 2:
        save_postfix += '_poss_err_full'
    elif args.add_pose_err == 3:
        save_postfix += '_poss_pred_det3d'

    if hpams['arch'] == 'supnerf':
        save_postfix = f'{save_postfix}_reg_iters_{args.reg_iters}'

    if 'pred_wlh' in hpams['net_hyperparams'].keys() and hpams['net_hyperparams']['pred_wlh'] > 0 and args.pred_wlh:
        save_postfix += f'_pred_wlh{args.pred_wlh}'

    if args.pred_box2d:
        save_postfix += '_pred_box2d'

    # if 'trainval' in hpams['dataset']['test_nusc_version']:
    if 'trainval' in args.nusc_version:
        save_postfix += '_full_val'

    if args.num_subset != 1:
        save_postfix += f'_subset_{args.id_subset}_of_{args.num_subset}'

    optimizer = OptimizerNuScenes(args.gpu, nusc_dataset, hpams,
                                  model_epoch=args.model_epoch, code_level=code_level,
                                  opt_pose=args.opt_pose,
                                  reg_iters=args.reg_iters, 
                                  opt_multiview=args.opt_multiview,
                                  pred_wlh=args.pred_wlh,
                                  cross_eval_folder=args.cross_eval_folder,
                                  num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix, save_freq=1000, vis=args.vis)

    # run-time optimization
    if args.cross_eval_folder is None:
        optimizer.run()
    else:
        optimizer.save_dir = args.cross_eval_folder

    # conduct cross-view evaluation of rgb and depth (multi-view optimization does not need, just use mean)
    if not args.opt_multiview:
        cross_vis_iter = 50 if args.vis > 0 else None
        optimizer.eval_cross_view(vis_iter=cross_vis_iter)
        cross_eval_file = os.path.join(optimizer.cross_eval_folder, 'cross_eval.pth')
    else:
        cross_eval_file = None

    # eval summary
    if args.opt_pose > 0:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        result_file = os.path.join(optimizer.save_dir, 'codes+poses.pth')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        result_file = os.path.join(optimizer.save_dir, 'codes.pth')
    collect_eval_results(result_file, hpams['optimize']['num_opts'], axes, 'b', args.opt_pose, cross_eval_file)

    plt.tight_layout()
    plt.savefig(os.path.join(optimizer.save_dir, 'eval.pdf'), format="pdf")
    # plt.show()
