export PYTHONPATH="$PWD:$PYTHONPATH"
python scripts/eval_saved_result.py --model-folder checkpoints/supnerf --test-folder test_nuscenes --legend-name SUPNeRF-nuScenes --plot-cross-view
python scripts/eval_saved_result.py --model-folder checkpoints/autorfmix --test-folder test_nuscenes --legend-name AutoRF-nuScenes --plot-cross-view
python scripts/eval_saved_result.py --model-folder checkpoints/supnerf --test-folder test_kitti --legend-name SUPNeRF-KITTI
python scripts/eval_saved_result.py --model-folder checkpoints/autorfmix --test-folder test_kitti --legend-name AutoRF-KITTI
python scripts/eval_saved_result.py --model-folder checkpoints/supnerf --test-folder test_waymo --legend-name SUPNeRF-Waymo
python scripts/eval_saved_result.py --model-folder checkpoints/autorfmix --test-folder test_waymo --legend-name AutoRF-Waymo