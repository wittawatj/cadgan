#!/bin/bash

python3 run_gkmm.py \
    --extractor_type vgg_face \
    --extractor_layers 35 \
    --texture 0\
    --depth_process no \
    --g_path celebAHQ_00/chkpts/model.pt \
    --g_type celebAHQ.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_celeba_face/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 99 \
    --img_log_steps 500 \
    --cond_path  celebaHQ/ \
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
