import os
import matplotlib.pyplot as plt
import argparse
from src.utils import collect_eval_results
import sys


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-folder", dest="model_folder", default="checkpoints/supnerf")
    arg_parser.add_argument("--test-folder", dest="test_folder", default="test_nuscenes")
    arg_parser.add_argument("--legend-name", dest="legend_name", default="SUPNeRF-nuScenes")
    arg_parser.add_argument("--plot-cross-view", dest="plot_cross_view", action='store_true')
    arg_parser.add_argument("--manual-lim", dest="manual_lim", action='store_true')
    arg_parser.add_argument("--save-dir", dest="save_dir", default='eval_summary')
    args = arg_parser.parse_args()
    
    # log_file = os.path.join(args.save_dir, args.legend_name + '.txt')
    # sys.stdout = open(log_file, 'w')
    
    print(f'========================= Evaluating {args.legend_name} =========================')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_name = args.legend_name + '.pdf'
    save_file = os.path.join(args.save_dir, save_name)

    test_folders = [
        args.test_folder,
    ]
    legend_names = [
        args.legend_name,
    ]
    colors = ['m', 'g', 'b',  'k', 'c', 'r']
    print_iters = [0, 3, 5, 10, 20, 50, 99]
    max_iter = 100

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    print(f'iters: {print_iters}')
    # fig.suptitle(save_name[:-4])

    line00_holders = []
    line01_holders = []
    line10_holders = []
    line11_holders = []

    for i, test_folder in enumerate(test_folders):
        opt_pose = 'opt_pose' in test_folder
        result_file = f'{args.model_folder}/{test_folder}/codes+poses.pth'
        cross_eval_file = f'{args.model_folder}/{test_folder}/cross_eval.pth'
        if not args.plot_cross_view:
            cross_eval_file = None

        color = colors[i]
        lines = collect_eval_results(result_file, max_iter, axes, color, opt_pose, cross_eval_file, print_iters,
                                     rot_outlier_ignore=False)

        line00_holders.append(lines[0])
        line01_holders.append(lines[1])
        if opt_pose:
            line10_holders.append(lines[2])
            line11_holders.append(lines[3])

    # arrange all the legends
    axes[0, 0].legend(line00_holders, legend_names)
    axes[0, 1].legend(line01_holders, legend_names, loc='upper right')
    axes[1, 0].legend(line10_holders, legend_names, loc='upper right')
    axes[1, 1].legend(line11_holders, legend_names, loc='upper right')

    # use manual tuned y lim
    if args.manual_lim:
        axes[0, 1].set_ylim([0.5, 2.5])
        axes[1, 0].set_ylim([5.0, 10.0])
        axes[1, 1].set_ylim([0.5, 1.8])

        # axes[0, 1].set_ylim([0.8, 4])
        # axes[1, 0].set_ylim([6.0, 20.0])
        # axes[1, 1].set_ylim([0.9, 2.5])

    plt.tight_layout()
    plt.savefig(save_file, format="pdf")
    # plt.show()
