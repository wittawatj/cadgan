#!/bin/bash

python3 run_lars_gkmm_interpolation.py \
    --extractor_type vgg_face \
    --extractor_layers 8 17 26 35 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/celebAHQ_00/chkpts/model.pt \
    --g_type celebAHQ.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_celeba_vggface/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 3000 \
    --lr 1e-2 \
    --seed 9 \
    --img_log_steps 10 \
    --cond_path  c.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --w_intp 0
