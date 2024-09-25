import sys, os
ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))
import argparse
import json
import wandb
from datetime import date, datetime

from src.trainer_nerf_nuscenes import TrainerNerfNuscenes
from src.trainer_unified_nuscenes import TrainerUnifiedNuscenes
from src.data_nuscenes import NuScenesData


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpus", dest="gpus", type=int, default=4,
                            help='Number of GPUs to use')
    arg_parser.add_argument("--config_file", dest="config_file", default="jsonfiles/supnerf.nusc.vehicle.car.json")
    arg_parser.add_argument("--seg_source", dest="seg_source",
                            default='instance',
                            help="use predicted instance/panoptic segmentation on nuscenes dataset")
    arg_parser.add_argument("--pretrained_model_dir", dest="pretrained_model_dir", default=None,
                            help="location of saved pretrained model and codes")
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int, default=48)
    arg_parser.add_argument("--num_workers", dest="num_workers", type=int, default=16)
    arg_parser.add_argument("--epochs", dest="epochs", type=int, default=40)
    arg_parser.add_argument("--resume_from_epoch", dest="resume_from_epoch", default=None)
    arg_parser.add_argument("--resume_dir", dest="resume_dir", default=None)
    arg_parser.add_argument("--im_enc_rate", dest="im_enc_rate", type=float, default=1.0,
                            help="the rate to mix the use of encoder code and track-wise recode code")
    arg_parser.add_argument("--pred_box2d", dest="pred_box2d", type=int, default=0,
                            help="whether to use predicted 2d box from maskrcnn for optimization")
    arg_parser.add_argument("--aug_box2d", dest="aug_box2d", type=int, default=0,
                            help="whether to use augment 2d box for more diverse crop")
    arg_parser.add_argument("--aug_wlh", dest="aug_wlh", type=int, default=0,
                            help="whether to use augment 3d box wlh for more robustness to different test wlh")
    arg_parser.add_argument("--finetune_wlh", dest="finetune_wlh", type=int, default=0,
                            help="if the training finetune wlh, training apply wlh loss")
    arg_parser.add_argument("--render_sz", dest="render_sz", type=int, default=None,
                            help="If not None, image will be resized to get tgt rgb value for the nerf loss")
    args = arg_parser.parse_args()

    # Read Hyper-parameters
    with open(args.config_file, 'r') as f:
        hpams = json.load(f)

    nusc_data_dir = hpams['dataset']['train_data_dir']
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_' + args.seg_source)
    nusc_version = hpams['dataset']['train_nusc_version']
    save_dir = hpams['dataset'][
                   'nusc_cat'] + '.' + nusc_version + '.use_' + args.seg_source + f'.bsize{args.batch_size}'

    if 'autorf' in args.config_file or 'supnerf' in args.config_file:
        if hpams['net_hyperparams']['norm_layer_type'] == 'InstanceNorm2d':
            save_dir += '.insnorm'
        save_dir += f'.e_rate{args.im_enc_rate}'
    
    if args.pred_box2d:
        save_dir += '_pred_box2d'
        
    if args.aug_box2d:
        save_dir += '_aug_box2d'
        
    if args.aug_wlh:
        save_dir += '_aug_wlh'
        
    if args.finetune_wlh:
        save_dir += '_finetune_wlh'
        if hpams['net_hyperparams']['pred_wlh'] == 0:
            print('ERROR: network arch does not support wlh finetune')
            sys.exit()

    today = date.today()
    log_name = os.path.basename(save_dir) + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    dt_str = today.strftime('_%Y_%m_%d')
    save_dir += dt_str

    # create dataset
    nusc_dataset = NuScenesData(
        hpams,
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='train',
        out_gt_depth=False,
        add_pose_err=2,
        pred_box2d=args.pred_box2d,
        aug_box2d=args.aug_box2d,
        render_sz=args.render_sz,
        prepare_batch_rays=True
    )

    if 'supnerf' in args.config_file:
        # create trainer
        trainer = TrainerUnifiedNuscenes(save_dir, args.gpus, nusc_dataset,
                                         args.pretrained_model_dir, 
                                         args.resume_from_epoch, args.resume_dir,
                                         args.config_file,
                                         args.batch_size, num_workers=args.num_workers, shuffle=True,
                                         im_enc_rate=args.im_enc_rate, aug_box2d=args.aug_box2d, aug_wlh=args.aug_wlh,
                                         finetune_wlh=args.finetune_wlh)

    else:
        # create trainer
        trainer = TrainerNerfNuscenes(save_dir, args.gpus, nusc_dataset,
                                      args.pretrained_model_dir, 
                                      args.resume_from_epoch, args.resume_dir,
                                      args.config_file,
                                      args.batch_size, num_workers=args.num_workers, shuffle=True,
                                      im_enc_rate=args.im_enc_rate)

    # wandb.init(project="SUPNeRF", entity="33yuliangguo", config=hpams, name=log_name)
    
    # training
    trainer.train(args.epochs)
