#!/bin/bash

python3 run_lars_gkmm.py \
    --extractor_type hed_color \
    --texture 0\
    --depth_process no \
    --g_path gan_data/celebAHQ_00/chkpts/model.pt \
    --g_type celebAHQ.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_celeba_hed/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 3000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 500 \
    --cond_path  celebahq_imgs/c.png \
    --kernel imq \
    --kparams -0.5 1e+2 \
