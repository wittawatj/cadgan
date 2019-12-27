#!/bin/bash

python3 run_gkmm.py \
    --extractor_type mnist_cnn_digit_layer_color \
    --extractor_layers 3 \
    --texture 0\
    --depth_process no \
    --g_path dcgan_colormnist/colormnist/netG_epoch_24.pth \
    --g_type colormnist_dcgan \
    --g_min 0 \
    --g_max 1.0 \
    --logdir log_mnist_color/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 1e-3 \
    --seed 99 \
    --img_log_steps 10000 \
    --cond_path colormnist/ \
    --kernel imq \
    --kparams -0.5 1e+1 \
    --img_size 28
