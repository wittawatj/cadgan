#!/bin/bash

python3 run_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 4 \
    --texture 0\
    --depth_process no \
    --g_path lsun_bridge/chkpts/model.pt \
    --g_type lsun_bridge.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bridge/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 3000 \
    --lr 1e-1 \
    --seed 99 \
    --img_log_steps 500 \
    --cond_path  lsun_bridge/\
    --kernel imq \
    --kparams -0.5 1e+2 \
