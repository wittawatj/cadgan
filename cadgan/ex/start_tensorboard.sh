#!/bin/bash

# One input argument: Tensorboard log folder
python3 -m tensorboard.main --logdir $1
