#!/bin/bash

python3 run_gkmm.py \
    --extractor_type mnist_cnn_digit_layer \
    --extractor_layers 2 \
    --texture 0\
    --depth_process no \
    --g_path prob_models/mnist_dcgan/mnist_dcgan_ep40_bs64.pt \
    --g_type mnist_dcgan \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_rebuttal_mnist/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 1e-1 \
    --seed 9 \
    --img_log_steps 10000 \
    --cond_path b.txt \
    --kernel imq \
    --kparams -0.5 1e+1 \
    --img_size 28
