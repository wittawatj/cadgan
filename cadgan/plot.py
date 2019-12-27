"""Module containing convenient functions for plotting"""

from builtins import object, range

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision

__author__ = "wittawat"


def set_default_matplotlib_options():
    # font options
    font = {
        #'family' : 'normal',
        #'weight' : 'bold',
        "size": 18
    }

    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def show_torch_imgs(imgs, nrow=8, figsize=(8, 5), **opt):
    """
    A convenient function to show a stack of images (Pytorch tensors).
    """
    # https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91
    img = torchvision.utils.make_grid(imgs, nrow=nrow, **opt)
    npimg = img.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
