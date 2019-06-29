""" 
Runnable script to experiment with the kernel moment matching with a
generator (conditional image generation with kernel moment matching). Pytorch
version. Data = Mnist.
"""

__author__ = "wittawat"

import argparse
import copy
import datetime
# import matplotlib
# import matplotlib.pyplot as plt
import os
import pprint
import sys

# KBRGAN
import cadgan
import cadgan.gen as gen
import cadgan.glo as glo
import cadgan.kernel as kernel
import cadgan.log as log
# import cadgan.plot as plot
# import cadgan.embed as embed
# import cadgan.util as util
import cadgan.main as kmain
import cadgan.net.net as net
import cadgan.util as util
import dill
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from cadgan.mnist.classify import MnistClassifier
from tensorboardX import SummaryWriter


def sample_by_labels(data, label_counts):
    """
    data: a dataset such that data[i][0] is a point, and data[i][1] is an integer label.
    label_counts: a list of tuples of two values (A, B), where A is a label, and B is the count.

    Return a Pytorch stack of data selected according to label_counts.
    """
    list_selected = []
    labels = np.array([data[i][1] for i in range(len(data))])
    for label, count in label_counts:
        inds = np.where(labels == label)[0]
        homo_data = [data[i][0] for i in inds[:count]]
        list_selected.extend(homo_data)
    # stack all
    selected = torch.stack(list_selected)
    return selected


# def pt_gkmm_mnist(g, cond_imgs, extractor, k, Z, optimizer,
#         device=torch.device('cpu'), tensor_type=torch.FloatTensor,
#         n_opt_iter=500, seed=1):
#     pass


def get_data(data_folder, train=False):
    # load MNIST data
    # data_folder = glo.data_file('mnist')
    mnist_dataset = torchvision.datasets.MNIST(
        data_folder,
        train=train,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    return mnist_dataset


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch GKMM on MNIST. Some paths are relative to the "(share_path)/prob_models/". See settings.ini for (share_path).'
    )

    parser.add_argument(
        "--extractor_path",
        type=str,
        help="Relative path \
        (relative to (share_path)/prob_models) to the file that can be loaded with \
        torch.load(). The saved object should be a torch.nn.Module representing a \
        feature extractor. ",
        required=True,
    )
    parser.add_argument(
        "--g_path",
        type=str,
        required=True,
        help="Relative path \
            (relative to (share_path)/prob_models) to the file that can be loaded \
            to get a cadgan.gen.PTNoiseTransformer representing an image generator.",
    )
    parser.add_argument(
        "--logdir", type=str, required=True, help="full path to the folder to contain Tensorboard log files"
    )
    parser.add_argument(
        "--device", nargs="?", choices=["cpu", "gpu"], default="cpu", help="Device to use for computation."
    )
    parser.add_argument("--n_sample", type=int, default=16, metavar="n", help="Number of images to generate")
    parser.add_argument("--n_opt_iter", type=int, default=500, help="Number of optimization iterations")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (for the optimizer)")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="Random seed. Among others, this affects the initialization of the noise vectors of the generator in the optimization.",
    )
    parser.add_argument(
        "--img_log_steps",
        type=int,
        default=10,
        metavar="N",
        help="how many optimization iterations to wait before logging generated images",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="mnist/",
        help="Relative path (relative to the data folder) \
            containing Mnist training data. Mnist data will be downloaded if \
            not existed already.",
    )
    parser.add_argument(
        "--cond",
        nargs="+",
        type=int,
        dest="cond",
        action="append",
        required=True,
        help='Digit label and number of images from that label to condition on. For example, "--cond 3 4" means 4 images of digit 3. --cond can be used multiple times. For instance, use --cond 1 2 --cond 3 1 to condition on 2 digits of 1, and 1 digit of 3',
    )
    parser.add_argument(
        "--kernel",
        nargs="?",
        required=True,
        choices=["linear", "gauss", "imq"],
        help="choice of kernel to put on top of extracted features.  May need to specify also --kparams.",
    )
    parser.add_argument(
        "--kparams",
        nargs="*",
        type=float,
        dest="kparams",
        default=[],
        help="A list of kernel parameters (float). Semantic of parameters depends on the chosen kernel",
    )

    # glo.data_file('mnist/')

    args = parser.parse_args()
    print("Training options: ")
    args_dict = vars(args)
    pprint.pprint(args_dict, width=5)

    # ---------------------------------

    # True to use GPU
    use_cuda = args.device == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)

    # load option depends on whether GPU is used
    device_load_options = {} if use_cuda else {"map_location": lambda storage, loc: storage}

    # check --cond arguments. This is a list of lists.
    label_counts = args.cond
    for each in label_counts:
        if len(each) != 2:
            raise ValueError("Invalid cond argument. Must be a list of 2 integers. Found {}".format(each))

    # def pt_gkmm(g, cond_imgs, extractor, k, Z, optimizer,
    #         device=torch.device('cpu'), tensor_type=torch.FloatTensor,
    #         n_opt_iter=500, seed=1):

    # load generator
    g_fpath = glo.share_path("prob_models", args.g_path)
    log.l().info("Loading the generator from {}".format(g_fpath))
    g = net.SerializableModule.load(g_fpath, **device_load_options)
    assert isinstance(g, gen.PTNoiseTransformer)

    # full path to the saved feature extractor file
    extractor_fpath = glo.share_path("prob_models", args.extractor_path)
    log.l().info("Loading the feature extractor from {}".format(extractor_fpath))
    extractor = torch.load(extractor_fpath, pickle_module=dill, **device_load_options)
    assert isinstance(extractor, torch.nn.Module)

    # full path to the data (Mnist) folder
    data_dir_path = glo.data_file(args.data_dir)
    mnist_data = get_data(data_dir_path)

    # get the images to condition on. This is deterministic.
    cond_imgs = sample_by_labels(mnist_data, label_counts)
    cond_imgs = cond_imgs.to(device).type(tensor_type)

    # initialize the noise vectors for the generator
    # Set the random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_sample = args.n_sample
    # sample a stack of noise vectors
    Z = g.sample_noise(n_sample)

    # kernel on top of the extracted features
    k_map = {"linear": kernel.PTKLinear, "gauss": kernel.PTKGauss, "imq": kernel.PTKIMQ}
    kernel_key = args.kernel
    kernel_params = args.kparams
    k_constructor = k_map[kernel_key]
    # construct the chosen kernel with the specified parameters
    k = k_constructor(*kernel_params)

    # optimizer
    lr = args.lr
    optimizer = torch.optim.Adam([Z], lr=lr)
    n_opt_iter = args.n_opt_iter
    logdir = args.logdir
    img_log_steps = args.img_log_steps

    # dictionary containing key-value pairs for experimental settings.
    log_str_dict = dict((ke, str(va)) for (ke, va) in args_dict.items())

    # logdir is just a parent folder.
    # Form the actual file name by concatenating the values of all
    # hyperparameters used.
    log_str_dict2 = copy.deepcopy(log_str_dict)

    util.translate_keys(
        log_str_dict2,
        {
            "cond": "co",
            "data_dir": "dat",
            "extractor_path": "ext",
            "g_path": "g",
            "kernel": "k",
            "kparams": "kp",
            "n_opt_iter": "it",
            "n_sample": "n",
            "seed": "s",
        },
    )

    parameters_str = util.dict_to_string(
        log_str_dict2, exclude=["device", "img_log_steps", "log_dir"], entry_sep="-", kv_sep="_"
    )
    logdir_fname = util.clean_filename(parameters_str, replace="/\\[]")
    log_dir_path = os.path.join(logdir, logdir_fname)

    # run the kernel moment matching optimization
    log.l().info("Saving this run to: {}".format(log_dir_path))
    kmain.pt_gkmm(
        g,
        cond_imgs,
        extractor,
        k,
        Z,
        optimizer,
        log_dir_path=log_dir_path,
        log_str_dict=log_str_dict,
        device=device,
        tensor_type=tensor_type,
        n_opt_iter=n_opt_iter,
        seed=seed,
        img_log_steps=img_log_steps,
    )


if __name__ == "__main__":
    main()
