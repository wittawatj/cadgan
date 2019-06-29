#!/bin/bash

python run_pt_gkmm_mnist.py \
    --extractor_path mnist_cnn/mnist_l2_cnn.pt \
    --g_path mnist_dcgan/ptnt_mnist_dcgan_ep40_bs64.pt \
    --logdir log_dcgan_mnist/ \
    --device gpu \
    --n_sample 8 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 3 \
    --img_log_steps 10 \
    --data_dir mnist/ \
    --cond 1 2 --cond 7 2 \
    --kernel linear \
    #--kparams -0.5 1e+2 \
    #--kernel linear \
    #--kparams
