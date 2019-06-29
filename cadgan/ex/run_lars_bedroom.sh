#!/bin/bash

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b1.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b1.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 3 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b1.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b1.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 5 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b1.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b2.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b2.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 3 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b2.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b2.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 5 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b2.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b3.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b3.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 3 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b3.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b3.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 5 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b3.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b4.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b4.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 3 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b4.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b4.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 5 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b4.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 1 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b5.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
    
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 2 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b5.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 3 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b5.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224

python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 4 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b5.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224
python3 run_lars_gkmm.py \
    --extractor_type resnet50_365 \
    --extractor_layers 17 \
    --texture 0\
    --depth_process no \
    --g_path gan_data/lsun_bedrooms/chkpts/model.pt \
    --g_type lsun_bedroom.yaml \
    --g_min -1.0 \
    --g_max 1.0 \
    --logdir log_lars_bedroom_debug_minsky_layer_m3/ \
    --device gpu \
    --n_sample 5 \
    --n_opt_iter 1000 \
    --lr 5e-2 \
    --seed 9 \
    --img_log_steps 1000 \
    --cond_path  b5.txt\
    --kernel imq \
    --kparams -0.5 1e+2 \
    --img_size 224