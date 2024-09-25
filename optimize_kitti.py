import sys, os
ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))
import argparse
import json
import matplotlib.pyplot as plt
from src.utils import str2bool
from src.optimizer_kitti import OptimizerKitti
from src.data_kitti import KittiData
from src.utils import collect_eval_results


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpu", dest="gpu", type=int, default=0)
    # arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/autorfmix.kitti.car.json")
    arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/supnerf.kitti.car.json")
    arg_parser.add_argument("--model_epoch", dest="model_epoch", default=39, help="Specify certain epoch model to load")
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    arg_parser.add_argument("--opt_pose", dest="opt_pose", type=int, default=1,
                            help="0: given init pose, optimize codes only."
                                 "1: given init pose, optimize both pose and codes. "
                                 "2: direct predict pose from pnp and then optimize both pose and codes")
    arg_parser.add_argument("--reg_iters", dest="reg_iters", type=int, default=3,
                            help="number of iters to use dl regressor to update."
                                 "If 0, pose updates purely rely on Nerf BP")
    arg_parser.add_argument("--add_pose_err", dest="add_pose_err", type=int, default=2,
                            help="0: no error pose from dataset"
                                 "1: controlled pose err from dataset"
                                 "2: full range pose err in reasonable range"
                                 "3: use associated third-party 3d detection results with target mask")
    arg_parser.add_argument("--init_rot_err", dest="init_rot_err", type=float, default=0.4,
                            help="Apply initial error of rotation in radians, for add_pose_err=1")
    arg_parser.add_argument("--init_trans_err", dest="init_trans_err", type=float, default=0.01,
                            help="Apply initial error of translation in ratio of distance to object center, for add_pose_err=1")
    arg_parser.add_argument("--rand_angle_lim", dest="rand_angle_lim", type=float, default=0,
                            help="normally 0 for test case, np.pi/9 is normally used for training, for add_pose_err=1")
    arg_parser.add_argument("--pred_box2d", dest="pred_box2d", type=int, default=0,
                            help="whether to use predicted 2d box from maskrcnn for optimization")
    arg_parser.add_argument("--pred_wlh", dest="pred_wlh", type=int, default=0,
                            help="whether to use wlh of 3d box from nerf model")
    arg_parser.add_argument("--vis", dest="vis", type=int, default=0,
                            help="0: no image saved; 1: save start and end frame; 2: save all frames")
    arg_parser.add_argument("--num-samples2eval", dest="num_samples2eval", type=int, default=None,
                            help="if not None, do not eval later sample for a quicker test")
    args = arg_parser.parse_args()

    # Read Hyper-parameters
    with open(args.config_file, 'r') as f:
        hpams = json.load(f)

    hpams_pose_refiner = None
    hpams_pose_regressor = None

    kitti_data_dir = hpams['dataset']['data_dir']

    # create dataset
    kitti_dataset = KittiData(
        hpams,
        kitti_data_dir,
        split='val',
        debug=False,
        add_pose_err=args.add_pose_err,
        init_rot_err=args.init_rot_err,
        init_trans_err=args.init_trans_err,
        rand_angle_lim=args.rand_angle_lim,
        pred_box2d=args.pred_box2d
    )

    # create optimizer
    save_postfix = '_kitti'
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

    optimizer = OptimizerKitti(args.gpu, kitti_dataset, hpams,
                               model_epoch=args.model_epoch, opt_pose=args.opt_pose,
                               reg_iters=args.reg_iters, pred_wlh=args.pred_wlh,
                               num_workers=args.num_workers, shuffle=False, save_postfix=save_postfix, save_freq=1000,
                               vis=args.vis, num_samples2eval=args.num_samples2eval)

    # run-time optimization
    optimizer.run()

    # eval summary
    if args.opt_pose > 0:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        result_file = os.path.join(optimizer.save_dir, 'codes+poses.pth')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        result_file = os.path.join(optimizer.save_dir, 'codes.pth')

    collect_eval_results(result_file, hpams['optimize']['num_opts'], axes, 'b', args.opt_pose, None)

    plt.tight_layout()
    plt.savefig(os.path.join(optimizer.save_dir, 'eval.pdf'), format="pdf")
    plt.show()
