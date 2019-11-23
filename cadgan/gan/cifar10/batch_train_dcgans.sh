#!/bin/bash

#CIFAR10 classes and their indices:

#[(0, 'airplane'),
# (1, 'automobile'),
# (2, 'bird'),
# (3, 'cat'),
# (4, 'deer'),
# (5, 'dog'),
# (6, 'frog'),
# (7, 'horse'),
# (8, 'ship'),
# (9, 'truck')]

python dcgan.py --n_epochs 150 --batch_size 32 --sample_interval 100 --classes 0 --save_model_interval 30
python dcgan.py --n_epochs 150 --batch_size 32 --sample_interval 100 --classes 1 --save_model_interval 30
python dcgan.py --n_epochs 150 --batch_size 32 --sample_interval 100 --classes 8 --save_model_interval 30
python dcgan.py --n_epochs 150 --batch_size 32 --sample_interval 100 --classes 9 --save_model_interval 30

python dcgan.py --n_epochs 220 --batch_size 32 --sample_interval 100 --classes 1 9 --save_model_interval 30
python dcgan.py --n_epochs 220 --batch_size 32 --sample_interval 100 --classes 0 8 --save_model_interval 30
python dcgan.py --n_epochs 150 --batch_size 32 --sample_interval 100 --classes 0 1 8 9 --save_model_interval 30
