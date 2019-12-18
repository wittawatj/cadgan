#!/bin/bash

python3 run_gkmm.py \
    --extractor_type pixel \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path CUB_data/model.pt \
    --g_type cub.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir CUB_pixel_experiment/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 3000 \
    --lr 1e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  test_list/cub_n1/files_n1_95.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
