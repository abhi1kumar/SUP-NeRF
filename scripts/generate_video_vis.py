import os
import glob


if __name__ == '__main__':
    result_path = 'checkpoints/supnerf/test_nuscenes_opt_pose_1_poss_err_full_reg_iters_3_epoch_39_vis2/*'
    tgt_paths = glob.glob(result_path)

    for tgt_path in tgt_paths:
        if os.path.isfile(tgt_path):
            continue

        cmd = f'ffmpeg -r 20 -f image2 -i {tgt_path}/opt%03d.png -pix_fmt yuv420p {tgt_path}.avi'
        print(cmd)
        os.system(cmd)
