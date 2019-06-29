# import matplotlib
# import matplotlib.pyplot as plt
import os

import cadgan
import cadgan.embed as embed
import cadgan.fashion_mnist.dcgan as fashion_dcgan
import cadgan.kernel as kernel
import cadgan.main as main
import cadgan.util as util
import numpy as np
import scipy.stats as stats
import torch
import torchvision

options = {"n_epochs": 40, "batch_size": 2 ** 6}
dcgan = fashion_dcgan.DCGAN(**options)

model_fname = "fashion_dcgan_ep{}_bs{}.pt".format(options["n_epochs"], options["batch_size"])
model_fpath = os.path.join(dcgan.prob_model_dir, model_fname)

dcgan.train()
