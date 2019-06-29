"""
Runnable script to experiment with the kernel moment matching with a
generator (conditional image generation with kernel moment matching). Pytorch
version. Data = Mnist.
"""

__author__ = "wittawat"

import argparse
import copy
import datetime
import glob
# import matplotlib
# import matplotlib.pyplot as plt
import os
import pprint
import warnings

# KBRGAN
import cadgan
import cadgan.gen as gen
import cadgan.glo as glo
import cadgan.imutil as imutil
import cadgan.kernel as kernel
import cadgan.log as log
# import cadgan.plot as plot
# import cadgan.embed as embed
# import cadgan.util as util
import cadgan.main as kmain
import cadgan.net.extractor as ext
import cadgan.net.net as net
import cadgan.util as util
import dill
# LarsGAN
# GAN
import ganstab.configs
import imageio
import numpy as np
import scipy.misc
import skimage
import torch
import torchsummary
import torchvision
import torchvision.transforms as transforms
from cadgan.mnist.classify import MnistClassifier
from ganstab.gan_training import utils
# gan_training package is originally from https://github.com/LMescheder/GAN_stability
# Install it via https://github.com/wittawatj/GAN_stability
# - clone the repository
# - go to the folder. Run "pip install -e ."
# - You should be able to "import ganstab"
from ganstab.gan_training.checkpoints import CheckpointIO
from ganstab.gan_training.config import build_generator, load_config
from ganstab.gan_training.distributions import get_ydist, get_zdist
from ganstab.gan_training.eval import Evaluator
from tensorboardX import SummaryWriter
from torch import nn


def target_transform():
    return transforms.Compose([transforms.ToTensor()])


class Generator(nn.Module):
    def __init__(self, ydist, generator):
        super(Generator, self).__init__()
        self.ydist = ydist
        self.generator = generator

    def forward(self, Z):
        n_sample = Z.shape[0]
        ysam = self.ydist.sample((n_sample,))
        return self.generator(Z, ysam)


def main():
    warnings.warn(
        "This script is obsolete. Use run_lars_gkmm.py instead. This file will be deleted soon. Let Wittawat know if you have a good reason to keep."
    )

    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch GKMM on MNIST. Some paths are relative to the "(share_path)/prob_models/". See settings.ini for (share_path).'
    )

    parser.add_argument(
        "--extractor_type",
        type=str,
        default="vgg",
        help="The feature extractor, currently VGG is supported only. The saved object should be a torch.nn.Module representing a \
        feature extractor. ",
        required=True,
    )
    parser.add_argument(
        "--extractor_layers", nargs="+", default=["4", "9", "18", "27"], help="vgg layers for texture. Default:[]"
    )
    parser.add_argument("--texture", type=float, default=0, help="Use texture of extracted features. Default=0")

    parser.add_argument(
        "--depth_process",
        nargs="?",
        choices=["avg", "max", "no"],
        default="no",
        help="Processing module to run on the output from \
            each filter in the specified layer(s).",
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
        "--g_type", type=str, default="celebAHQ.yaml", help="Generator type based on the data it is trained for."
    )
    parser.add_argument(
        "--g_min", type=float, help="The minimum value of the pixel output from the generator.", required=True
    )
    parser.add_argument(
        "--g_max", type=float, help="The maximum value of the pixel output from the generator.", required=True
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
    # parser.add_argument('--data_dir', type=str,
    #        default='mnist/', help='Relative path (relative to the data folder) \
    #        containing Mnist training data. Mnist data will be downloaded if \
    #        not existed already.')
    # parser.add_argument('--cond', nargs='+', type=int, dest='cond',
    #        action='append', required=True, help='Digit label and number of images from that label to condition on. For example, "--cond 3 4" means 4 images of digit 3. --cond can be used multiple times. For instance, use --cond 1 2 --cond 3 1 to condition on 2 digits of 1, and 1 digit of 3')
    parser.add_argument("--cond_path", type=str, required=True, help="Path to imgs for conditioning")
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

    img_transform = target_transform()
    # glo.data_file('mnist/')
    args = parser.parse_args()
    print("Training options: ")
    args_dict = vars(args)
    pprint.pprint(args_dict, width=5)

    # ---------------------------------

    # Check if texture and extractor are called correctly
    if args.texture and not args.extractor_layers or args.texture and not args.extractor_type:
        parser.error("Texture call, Extractor layers and Extractor type must be given at the same time!")

    # True to use GPU
    use_cuda = args.device == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)

    # load option depends on whether GPU is used
    device_load_options = {} if use_cuda else {"map_location": lambda storage, loc: storage}

    # Loading Configs for LarsGAN
    yaml_folder = os.path.dirname(ganstab.configs.__file__)
    yaml_config_path = os.path.join(yaml_folder, args.g_type)
    config = load_config(yaml_config_path)

    # load generator
    nlabels = config["data"]["nlabels"]
    out_dir = config["training"]["out_dir"]
    checkpoint_dir = os.path.join(out_dir, "chkpts")

    generator = build_generator(config)

    # Put models on gpu if needed
    with torch.enable_grad():  # use_cuda??????
        generator = generator.to(device)

    # Use multiple GPUs if possible
    generator = nn.DataParallel(generator)
    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator)
    # Test generator
    if config["test"]["use_model_average"]:
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    # Loading Generator
    ydist = get_ydist(nlabels, device=device)

    full_g_path = glo.share_path(args.g_path)
    if not os.path.exists(full_g_path):
        raise ValueError("Generator file does not exist: {}".format(full_g_path))
    it = checkpoint_io.load(full_g_path)
    # initialize the noise vectors for the generator
    # Set the random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_sample = args.n_sample
    # sample a stack of noise vectors
    latent_dim = 256
    f_noise = lambda n: torch.randn(n, latent_dim).float()
    Z0 = f_noise(n_sample)

    # Noise distribution is Gaussian. Unlikely that the magnitude of the
    # coordinate is above the bound.
    z_penalty = kmain.TPNull()  # .TPSymLogBarrier(bound=4.2, scale=1e-4)
    args_dict["zpen"] = z_penalty

    # output range of the generator (according to what the user specifies)
    g_range = (args.g_min, args.g_max)

    # Sanity check. Check that the specified g-range is plausible
    g_out_uncontrolled = Generator(ydist, generator_test.to(device))
    temp_sample = g_out_uncontrolled.forward(Z0)
    kmain.pixel_values_check(temp_sample, g_range, "Generator's samples")

    # transform the output range of g to (0,1)
    g = nn.Sequential(
        g_out_uncontrolled,
        gen.LinearRangeTransform(from_range=g_range, to_range=(0, 1)),
        nn.Upsample(size=(224, 224), mode="bilinear"),
    )

    # Loading Extractor
    if args.extractor_type == "vgg":
        extractor_layers = [int(i) for i in args.extractor_layers]
        depth_process_map = {"no": ext.Identity(), "avg": ext.GlobalAvgPool()}
        post_process = depth_process_map[args.depth_process]
        extractor = ext.VGG19(layers=extractor_layers, layer_postprocess=post_process)
    elif args.extractor_type == "alexnet_365":
        extractor = ext.AlexNet_365()
    elif args.extractor_type == "resnet18_365":
        extractor = ext.ResNet18_365()
    elif args.extractor_type == "resnet50_365":
        extractor = ext.ResNet50_365()
    elif args.extractor_type == "resnet18_365Layer":
        extractor = ext.ResNet18_365Layer()
        weigh_logits = 1

    if use_cuda:
        extractor = extractor.cuda()
    # run a forward pass throught the extractor just to test
    assert isinstance(extractor, torch.nn.Module)

    print("Summary of the extractor:")
    try:
        torchsummary.summary(extractor, input_size=(3, 224, 224))
    except:
        log.l().info("Exception occured when getting a summary of the extractor")
    tmp_extracted = extractor(g(Z0[[0]]))
    n_features = torch.prod(torch.tensor(tmp_extracted.shape))
    print("Number of extracted features = {}".format(n_features))
    del tmp_extracted

    if not os.path.isdir(glo.share_path(args.cond_path)):  #
        #
        loaded = imutil.load_resize_image(glo.share_path(args.cond_path)).copy()
        cond_img = img_transform(loaded).unsqueeze(0).type(tensor_type)  # .to(device)

    else:
        # using all images in the folder

        list_imgs = glob.glob(glo.share_path(args.cond_path) + "*")

        for path_img in list_imgs:
            loaded = imutil.load_resize_image(path_img).copy()
            cond_img = img_transform(loaded).unsqueeze(0).type(tensor_type)  # .to(device)

            try:
                cond_imgs = torch.cat((cond_imgs.clone(), cond_img))
            except NameError:
                cond_imgs = cond_img.clone()
        # import pdb; pdb.set_trace()
    cond_imgs = cond_imgs.to(device).type(tensor_type)

    # kernel on top of the extracted features
    k_map = {"linear": kernel.PTKLinear, "gauss": kernel.PTKGauss, "imq": kernel.PTKIMQ}
    kernel_key = args.kernel
    kernel_params = args.kparams
    k_constructor = k_map[kernel_key]
    # construct the chosen kernel with the specified parameters
    k = k_constructor(*kernel_params)

    # texture flag
    texture = args.texture
    # run the kernel moment matching optimization
    n_opt_iter = args.n_opt_iter
    logdir = args.logdir
    print("LOGDIR: ", logdir)

    # dictionary containing key-value pairs for experimental settings.
    log_str_dict = dict((ke, str(va)) for (ke, va) in args_dict.items())

    # logdir is just a parent folder.
    # Form the actual file name by concatenating the values of all
    # hyperparameters used.
    log_str_dict2 = copy.deepcopy(log_str_dict)

    now = datetime.datetime.now()
    time_str = "{:02}.{:02}.{}_{:02}{:02}{:02}".format(now.day, now.month, now.year, now.hour, now.minute, now.second)
    log_str_dict2["t"] = time_str
    util.translate_keys(
        log_str_dict2,
        {
            "cond_path": "co",
            "data_dir": "dat",
            "depth_process": "dp",
            "extractor_path": "ep",
            "extractor_type": "et",
            "g_path": "gp",
            "g_type": "gt",
            "kernel": "k",
            "kparams": "kp",
            "n_opt_iter": "it",
            "n_sample": "n",
            "seed": "s",
            "texture": "te",
        },
    )

    parameters_str = util.dict_to_string(
        log_str_dict2, exclude=["device", "img_log_steps", "logdir", "g_min", "g_max"], entry_sep="-", kv_sep="_"
    )
    img_log_steps = args.img_log_steps
    logdir_fname = util.clean_filename(parameters_str, replace="/\\[]")
    log_dir_path = os.path.join(logdir, logdir_fname)

    # multiple restarts to refine the drawn Z. This is just a heuristic
    # so we start (hopefully) from a good initial point.
    k_img = kernel.PTKFuncCompose(k, f=extractor)
    # multi_restarts_refiner = kmain.ZRMMDMultipleRestarts(
    #         g, z_sampler=f_noise, k=k_img, X=cond_imgs,
    #         n_restarts=100,
    #         n_sample=Z0.shape[0],
    #         )
    multi_restarts_refiner = kmain.ZRMMDIterGreedy(
        g, z_sampler=f_noise, k=k_img, X=cond_imgs, n_draws=40, n_sample=Z0.shape[0]  # number of times to draw each z_i
    )

    sum_writer = SummaryWriter(log_dir=log_dir_path)
    # write all key-value pairs in log_str_dict
    for ke, va in log_str_dict.items():
        sum_writer.add_text(ke, va)

        # Get a better Z
    Z = multi_restarts_refiner(Z0)

    # optimizer
    lr = args.lr
    optimizer = torch.optim.Adam([Z], lr=lr)
    # optimizer = torch.optim.LBFGS([Z])  --> LBFGS doesn't really converge, we could try other optimizer as well
    # Solve the kernel moment matching problem
    kmain.pt_gkmm(
        g,
        cond_imgs,
        extractor,
        k,
        Z,
        optimizer,
        z_penalty=z_penalty,
        sum_writer=sum_writer,
        device=device,
        tensor_type=tensor_type,
        n_opt_iter=n_opt_iter,
        seed=seed,
        texture=texture,
        img_log_steps=img_log_steps,
        weigh_logits=0,
    )


if __name__ == "__main__":
    main()
