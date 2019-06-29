#!/bin/bash

python3 run_lars_gkmm_lsun_scenes_multi_input.py \
    --extractor_type resnet50_365 \
    --extractor_layers 4 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bridge/chkpts/model.pt \
    --g_type lsun_bridge.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bridge_debug/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 3000 \
    --lr 8e-2 \
    --seed 9 \
    --img_log_steps 10 \
    --cond_path  lsun_imgs/lsun_bridge/\
    --kernel imq \
    --kparams -0.5 1e+2 \
