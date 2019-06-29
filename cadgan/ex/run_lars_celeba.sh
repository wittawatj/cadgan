#!/bin/bash

python3 run_lars_gkmm.py \
    --extractor_type vgg_face \
    --extractor_layers 35 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/celebAHQ_00/chkpts/model.pt \
    --g_type celebAHQ.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir face_interpolation_test/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  /notebooks/psangkloy3/gdrive/condgan_share/interpolate.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --w_input 0.3 0.3 0.4 \
    --img_size 224
