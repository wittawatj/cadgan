#!/bin/bash

srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.4_0.4_0.2.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.25_0.5_0.25.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.2_0.4_0.4.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.4_0.2_0.4.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.5_0.25_0.25.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.5_0_0.5.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.5_0.5_0.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0_0.5_0.5.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.33_0.33_0.33.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0.25_0.25_0.5.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0_0_1.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "0_1_0.sh" &
srun -p long --gres=gpu:1 --exclude bender,walle bash run_docker_kbrgan_script.sh "/notebooks/psangkloy3/kbrgan_temp/kbrgan/ex/interpolation/" "1_0_0.sh" &
