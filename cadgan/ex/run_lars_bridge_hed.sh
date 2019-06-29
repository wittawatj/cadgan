#!/bin/bash

python3 run_lars_gkmm.py \
    --extractor_type hed \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bridge/chkpts/model.pt \
    --g_type lsun_bridge.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bridge_hed/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 3000 \
    --lr 5e-1 \
    --seed 9 \
    --img_log_steps 10 \
    --cond_path  lsun_imgs/lsun_bridge/e.jpg \
    --kernel linear 
