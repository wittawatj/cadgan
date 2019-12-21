"""
Runnable script to experiment with the kernel moment matching with a
generator (conditional image generation with kernel moment matching). Pytorch
version.
"""

__author__ = "wittawat, patsorn"

import argparse
import copy
import datetime
import glob
import os
import pprint

# Just don't use imageio. Use skimage.
# import imageio

import cadgan
import cadgan.gan.colormnist.dcgan as cmnist_dcgan
import cadgan.gen as gen
import cadgan.glo as glo
import cadgan.imutil as imutil
import cadgan.kernel as kernel
import cadgan.log as log
import cadgan.main as kmain
import cadgan.gan.mnist.dcgan as mnist_dcgan
import cadgan.gan.mnist.util as mnist_util
import cadgan.net.extractor as ext
import cadgan.net.net as net
import cadgan.util as util
import dill

# LarsGAN
# GAN
import ganstab.configs
import numpy as np
import scipy.misc
import skimage
import torch
import torchsummary
import torchvision
import torchvision.transforms as transforms
from ganstab.gan_training import utils
from cadgan.gan.mnist.classify import MnistClassifier
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
    def __init__(self, generator, ydist=None):
        super(Generator, self).__init__()
        self.ydist = ydist
        self.generator = generator

    def forward(self, Z):
        n_sample = Z.shape[0]
        if self.ydist == None:
            return self.generator(Z)
        else:
            ysam = self.ydist.sample((n_sample,))
            return self.generator(Z, ysam)


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        device = input.device.type
        output_gram = torch.Tensor(a, b * b).to(device)
        for i in range(a):
            features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL

            # G = torch.mm(features, features.t()).div( b * c * d)  # compute the gram product
            # G = G.div( b * c * d)
            output_gram[i] = torch.mm(features, features.t()).div(b * c * d).view(b * b)
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return output_gram.to(device)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch GKMM. Some paths are relative to the "(share_path)/prob_models/". See settings.ini for (share_path).'
    )

    parser.add_argument(
        "--extractor_type",
        type=str,
        default="vgg",
        help="The feature extractor. The saved object should be a torch.nn.Module representing a \
        feature extractor. Currently support [vgg | vgg_face | alexnet_365 | resnet18_365 | resnet50_365 | hed | mnist_cnn | pixel]",
        required=True,
    )
    parser.add_argument(
        "--extractor_layers",
        nargs="+",
        default=["4", "9", "18", "27"],
        help="Number of layers to include. Only for VGG feature extractor. Default:[]",
    )
    parser.add_argument("--texture", type=float, default=0, help="Use texture (grammatrix) of extracted features. Default=0")
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
        "--n_init_resample", type=float, default=1, help="number of time to resample z for the heuristic"
    )
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
    parser.add_argument("--img_size", type=int, default=224, help="image size nxn default 256")
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

    parser.add_argument(
        "--w_input",
        nargs="+",
        default=[],
        help="weight of the input, must be equal to the number of cond images and sum to 1. if none specified, equal weights will be used.",
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

    # initialize the noise vectors for the generator
    # Set the random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_sample = args.n_sample

    if args.g_type.endswith(".yaml"):
        # sample a stack of noise vectors
        latent_dim = 256
        f_noise = lambda n: torch.randn(n, latent_dim).float()
        Z0 = f_noise(n_sample)

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

        # for celebA HQ generator,
        # if args.g_type == 'celebAHQ.yaml':
        #    generator.add_resize(args.img_size)

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
            #download lars pre-trained model file if not existed
            
            raise ValueError("Generator file does not exist: {}".format(full_g_path))
        it = checkpoint_io.load(full_g_path)

    elif args.g_type == "mnist_dcgan":
        # TODO should probablu reorganize these
        latent_dim = 100
        f_noise = lambda n: torch.randn(n, latent_dim).float()
        Z0 = f_noise(n_sample)

        full_g_path = glo.share_path(args.g_path)
        # load option depends on whether GPU is used
        load_options = {} if use_cuda else {"map_location": lambda storage, loc: storage}

        generator = mnist_dcgan.Generator() 
        if os.path.exists(full_g_path):
            generator.load_state_dict(torch.load(full_g_path, **load_options).state_dict(), strict=False)
        else:
            print("Generator file does not exist: {}\nLoading pretrain model...".format(full_g_path))
            generator.download_pretrain()  # .load(full_g_path, **load_options)
            
        generator = generator.to(device)

        generator_test = generator
        ydist = None

    elif args.g_type == "colormnist_dcgan":
        # TODO should probablu reorganize these
        latent_dim = 100
        f_noise = lambda n: torch.randn(n, latent_dim).float()
        Z0 = f_noise(n_sample)

        full_g_path = glo.share_path(args.g_path)
        if not os.path.exists(full_g_path):
            raise ValueError("Generator file does not exist: {}".format(full_g_path))
        # load option depends on whether GPU is used
        load_options = {} if use_cuda else {"map_location": lambda storage, loc: storage}
        generator = cmnist_dcgan.Generator()  # .load(full_g_path, **load_options)
        generator.load_state_dict(torch.load(full_g_path, **load_options), strict=False)
        generator = generator.to(device)

        generator_test = generator
        ydist = None

    # Noise distribution is Gaussian. Unlikely that the magnitude of the
    # coordinate is above the bound.
    z_penalty = kmain.TPNull()  # kmain.TPSymLogBarrier(bound=4.2, scale=1e-4)
    args_dict["zpen"] = z_penalty

    # output range of the generator (according to what the user specifies)
    g_range = (args.g_min, args.g_max)

    # Sanity check. Check that the specified g-range is plausible
    g_out_uncontrolled = Generator(ydist=ydist, generator=generator_test.to(device))

    temp_sample = g_out_uncontrolled.forward(Z0)
    kmain.pixel_values_check(temp_sample, g_range, "Generator's samples")

    extractor_in_size = args.img_size

    # transform the output range of g to (0,1)
    g = nn.Sequential(
        g_out_uncontrolled,
        nn.AdaptiveAvgPool2d((extractor_in_size, extractor_in_size)),
        gen.LinearRangeTransform(from_range=g_range, to_range=(0, 1)),
    )
    depth_process_map = {"no": ext.Identity(), "avg": ext.GlobalAvgPool()}
    feature_size = 128

    if args.texture == 1:
        post_process = nn.Sequential(depth_process_map[args.depth_process], GramMatrix())
    else:
        post_process = nn.Sequential(depth_process_map[args.depth_process])

    # Loading Extractor
    if args.extractor_type == "vgg":
        extractor_layers = [int(i) for i in args.extractor_layers]
        extractor = ext.VGG19(layers=extractor_layers, layer_postprocess=post_process)
    elif args.extractor_type == "vgg_face":
        extractor_layers = [int(i) for i in args.extractor_layers]
        extractor = ext.VGG19_face(layers=extractor_layers, layer_postprocess=post_process)
    elif args.extractor_type == "alexnet_365":
        extractor = ext.AlexNet_365()
    elif args.extractor_type == "resnet18_365":
        extractor = ext.ResNet18_365()
    elif args.extractor_type == "resnet50_365":
        extractor = ext.ResNet50_365(n_remove_last_layers=2, layer_postprocess=post_process)
    elif args.extractor_type == "hed":
        # extractor_in_size = 256
        extractor = ext.HED(device=device, resize=feature_size)
    elif args.extractor_type == "hed_color":
        #stacking feature from HED and tiny image to get both edge and color information
        hed = ext.HED(device=device, resize=feature_size)
        tiny = ext.TinyImage(device=device, grid_size=(10, 10))
        extractor = ext.StackModule(device=device, module_list=[hed, tiny], weights=[0.01, 0.99])
    elif args.extractor_type == "hed_vgg":
        #stacking feature from HED and vgg feature to get both edge and high level vgg information
        feature_size = 128
        hed = ext.HED(device=device, resize=feature_size)
        extractor_layers = [int(i) for i in args.extractor_layers]
        vgg = ext.VGG19(layers=extractor_layers, layer_postprocess=post_process)
        extractor = ext.StackModule(device=device, module_list=[hed, vgg], weights=[0.99, 0.01])
    elif args.extractor_type == "hed_color_vgg":
        #stacking feature from HED, tiny image, and vgg feature to get edge, color, and high level vgg information
        feature_size = 128
        hed = ext.HED(device=device, resize=feature_size)
        extractor_layers = [int(i) for i in args.extractor_layers]
        vgg = ext.VGG19(layers=extractor_layers, layer_postprocess=post_process)
        tiny = ext.TinyImage(device=device, grid_size=(10, 10))
        extractor = ext.StackModule(device=device, module_list=[hed, vgg, tiny], weights=[0.005, 0.005, 0.99])
    elif args.extractor_type == "color":
        extractor = ext.TinyImage(device=device, grid_size=(128, 128))
    elif args.extractor_type == "color_count":
        # to use with Waleed color mnist only:
        # the purpose is to count color based on the template, currently not working as expected.
        prototypes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.4, 0.2, 0]])
        extractor = ext.SoftCountPixels(prototypes=prototypes, gwidth2=0.3, device=device, tensor_type=tensor_type)
    elif args.extractor_type == "mnist_cnn":
        depth_process_map = {"no": ext.Identity(), "avg": ext.GlobalAvgPool()}
        if args.texture == 1:
            post_process = nn.Sequential(depth_process_map[args.depth_process], GramMatrix())
        else:
            post_process = nn.Sequential(depth_process_map[args.depth_process])
        extractor = ext.MnistCNN(
            device="cuda" if use_cuda else "cpu", layer_postprocess=post_process, layer=int(args.extractor_layers[0])
        )
    elif args.extractor_type == "mnist_cnn_digit_layer":
        #using the last layer of MNIST CNN (digit classification)
        depth_process_map = {"no": ext.Identity(), "avg": ext.GlobalAvgPool()}
        if args.texture == 1:
            post_process = nn.Sequential(depth_process_map[args.depth_process], GramMatrix())
        else:
            post_process = nn.Sequential(depth_process_map[args.depth_process])
        extractor = ext.MnistCNN(device="cuda" if use_cuda else "cpu", layer_postprocess=post_process, layer=3)
    elif args.extractor_type == "mnist_cnn_digit_layer_color":
        # using the last layer of MNIST CNN (digit classification) stacking with color information from tiny image
        depth_process_map = {"no": ext.Identity(), "avg": ext.GlobalAvgPool()}
        if args.texture == 1:
            post_process = nn.Sequential(depth_process_map[args.depth_process], GramMatrix())
        else:
            post_process = nn.Sequential(depth_process_map[args.depth_process])
        mnistcnn = ext.MnistCNN(device="cuda" if use_cuda else "cpu", layer_postprocess=post_process, layer=3)
        color = ext.MaxColor(device=device)
        extractor = ext.StackModule(device=device, module_list=[mnistcnn, color], weights=[1, 99])
    elif args.extractor_type == "pixel":
        #raw pixel as feature
        extractor = ext.Identity(flatten=True, slice_dim=0 if args.g_type == "mnist_dcgan" else None)
    else:
        raise ValueError("Unknown extractor type. Check --extractor_type")

    if use_cuda:
        extractor = extractor.cuda()
    assert isinstance(extractor, torch.nn.Module)

    print("Summary of the extractor:")
    try:
        torchsummary.summary(extractor, input_size=(3, extractor_in_size, extractor_in_size))
    except:
        log.l().info("Exception occured when getting a summary of the extractor")

    # run a forward pass throught the extractor just to test
    tmp_extracted = extractor(g(Z0[[0]]))
    n_features = torch.prod(torch.tensor(tmp_extracted.shape))
    print("Number of extracted features = {}".format(n_features))
    del tmp_extracted

    def load_multiple_images(list_imgs):
        for path_img in list_imgs:
            loaded = imutil.load_resize_image(path_img, extractor_in_size).copy()
            cond_img = img_transform(loaded).unsqueeze(0).type(tensor_type)  # .to(device)
            try:
                cond_imgs = torch.cat((cond_imgs.clone(), cond_img))
            except NameError:
                cond_imgs = cond_img.clone()
        return cond_imgs

    if not os.path.isdir(glo.share_path(args.cond_path)):  #
        # read list of imgs if it's a text file
        if args.cond_path.endswith(".txt"):
            img_txt_path = glo.share_path(args.cond_path)
            with open(img_txt_path, "r") as f:
                data = f.readlines()

            list_imgs = [glo.share_path(x.strip()) for x in data if len(x.strip()) != 0]
            if not list_imgs:
                raise ValueError("Empty list of images to condiiton. Make sure that {} is valid".format(img_txt_path))

            cond_imgs = load_multiple_images(list_imgs)
        else:
            path_img = glo.share_path(args.cond_path)
            loaded = imutil.load_resize_image(path_img, extractor_in_size).copy()
            cond_imgs = img_transform(loaded).unsqueeze(0).type(tensor_type)  # .to(device)
    else:
        # using all images in the folder
        list_imgs = glob.glob(glo.share_path(args.cond_path) + "*")
        cond_imgs = load_multiple_images(list_imgs)

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
            "extractor_layers": "el",
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
        log_str_dict2,
        exclude=["device", "img_log_steps", "logdir", "g_min", "g_max", "g_path", "t"],
        entry_sep="-",
        kv_sep="_",
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

    tmp_gen = g(Z0)
    assert tmp_gen.shape[-1] == extractor_in_size and tmp_gen.shape[-2] == extractor_in_size
    del tmp_gen

    if len(args.w_input) == 0:
        input_weights = None
    else:
        assert cond_imgs.shape[0] == len(args.w_input), "number of input weights must equal to number of input images"
        input_weights = torch.Tensor([float(x) for x in args.w_input], device=device).type(tensor_type)

    # A heuristic to pick good Z to start the optimization
    multi_restarts_refiner = kmain.ZRMMDIterGreedy(
        g,
        z_sampler=f_noise,
        k=k_img,
        X=cond_imgs,
        n_draws=int(
            args.n_init_resample
        ),  # number of times to draw each z_i --> set to 1 since I want to test the latent optimization,
        n_sample=Z0.shape[0],
        device=device,
        tensor_type=tensor_type,
        input_weights=input_weights,
    )

    # Summary writer for Tensorboard logging
    sum_writer = SummaryWriter(log_dir=log_dir_path)

    # write all key-value pairs in log_str_dict to the Tensorboard
    for ke, va in log_str_dict.items():
        sum_writer.add_text(ke, va)

    with open(os.path.join(log_dir_path, "metadata"), "wb") as f:
        dill.dump(log_str_dict, f)

    imutil.save_images(cond_imgs, os.path.join(log_dir_path, "input_images"))

    gens = g.forward(Z0)
    gens_cpu = gens.to(torch.device("cpu"))
    imutil.save_images(gens_cpu, os.path.join(log_dir_path, "prior_images"))
    del gens
    del gens_cpu
    # import pdb; pdb.set_trace()
    # Get a better Z
    Z = multi_restarts_refiner(Z0)

    # Try to plot (in Tensorboard) extracted features as images if possible
    try:
        # if args.extractor_type == 'hed':
        feat_out = extractor.forward(cond_imgs)
        # import pdb; pdb.set_trace()
        feature_size = int(np.sqrt(feat_out.shape[1]))
        feat_out = feat_out.view(feat_out.shape[0], 1, feature_size, feature_size)
        gens_cpu = feat_out.to(torch.device("cpu"))
        imutil.save_images(gens_cpu, os.path.join(log_dir_path, "input_feature"))
        arranged_init_imgs = torchvision.utils.make_grid(gens_cpu, nrow=2, normalize=True)
        sum_writer.add_image("Init_feature", arranged_init_imgs)
        del feat_out
    except:
        log.l().info("unable to plot feature as image")
    # if args.w_intp
    # import pdb; pdb.set_trace()

    imutil.save_images(cond_imgs, os.path.join(log_dir_path, "input_images"))

    # optimizer
    optimizer = torch.optim.Adam([Z], lr=args.lr)  # ,momentum=0.99,nesterov=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000], gamma=0.1)
    # optimizer = torch.optim.LBFGS([Z]) # --> LBFGS doesn't really converge, we could try other optimizer as well
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
        input_weights=input_weights,
        img_log_steps=img_log_steps,
        log_img_dir=log_dir_path,
    )


if __name__ == "__main__":
    main()
