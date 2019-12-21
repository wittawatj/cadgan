"""
Module containing feature extractors for images.
"""

import os

import cadgan
import cadgan.glo as glo
import cadgan.main as kmain
import cadgan.gan.mnist.classify as mnist_classify
import hed.networks
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from cadgan.gan.mnist.classify import MnistClassifier

class MnistCNN(nn.Module):
    def __init__(self, device="cpu", layer_postprocess=None, layer=1):
        # layer-> 1,2 after conv, 3 final fc layer
        super(MnistCNN, self).__init__()
        # load up model
        model_path = glo.share_path("prob_models/mnist_cnn/mnist_cnn_ep40_s1.pt")
        if os.path.exists(model_path):
            self.classifier = mnist_classify.MnistClassifier()
            self.classifier.load(model_path)
        else:
            self.classifier = mnist_classify.MnistClassifier(load=True)
        self.classifier = self.classifier.eval()
        self.classifier = self.classifier.to(device)

        self.normalize = NormalizeChannels2d([0.1307], [0.3081])
        self.color2grey = nn.AdaptiveAvgPool3d((1, 28, 28))

        self.layer_postprocess = layer_postprocess
        self.layer = layer
        self.device = device

    def forward(self, x):
        #
        layer = self.layer
        bs = x.shape[0]
        post_process = self.layer_postprocess

        if x.shape[1] == 3:  # if not grey scale image
            x = self.color2grey(x)
        # nn.AdaptiveAvgPool2d(

        x = self.normalize(x[:, 0:1, :, :])

        x = F.relu(F.max_pool2d(self.classifier.conv1(x), 2))
        x1 = post_process(x)
        x = F.relu(F.max_pool2d(self.classifier.conv2(x), 2))
        x2 = post_process(x)
        x = x.view(-1, 320)
        x = self.classifier.fc2(F.relu(self.classifier.fc1(x)))
        x = F.log_softmax(x, dim=1)
        x3 = post_process(x)

        if layer == 1:
            return x1.view(bs, -1)
        elif layer == 2:
            return x2.view(bs, -1)
        elif layer == 3:
            return x3.view(bs, -1)
        else:
            print("Invalid layer for Mnist CNN")
            return None


class HED(nn.Module):
    def __init__(self, device="cpu", resize=64):
        """
        wrapper class for HED network (see https://github.com/xwjabc/hed)
        
        """

        super(HED, self).__init__()

        net = hed.networks.HED(device, resize)
        self.hed = nn.DataParallel(net).to(device)
        # net.cuda()

        self.normalize_module = NormalizeChannels2d(
            mean=[122.67891434, 116.66876762, 104.00698793], std=[1.0, 1.0, 1.0]
        )  # todo check the correct value

        # self.normalize_module = NormalizeChannels2d(
        #        mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        checkpoint = torch.load(glo.share_path("extractor_models", "HED", "hed_checkpoint.pt"))
        self.hed.load_state_dict(checkpoint["net"])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")

        # minus statistic
        x = self.normalize_module(x * 255)
        output = self.hed.forward(x)[-1]

        return output.view(output.shape[0], -1)


class TinyImage(nn.Module):
    def __init__(self, device="cpu", grid_size=(5, 5)):
        super(TinyImage, self).__init__()
        tiny = nn.AdaptiveAvgPool2d(grid_size)
        self.tiny = nn.DataParallel(tiny).to(device)

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")

        # minus statistic
        output = self.tiny.forward(x)

        return output.view(output.shape[0], -1)


class SoftCountPixels(nn.Module):
    """
    For each image in the batch, construct a soft-count for each pixel protype
    in the list of p pixel prototypes. So if the batch size is b, then return a
    bxp matrix A where A[i,j] is in [0,1] specifying (roughly) the score 
    that prototype j occurs in image i.
    Soft count is done with an exponentiated p-norm kernel
    """

    def __init__(self, prototypes, gwidth2=0.3, pnorm=2, device="cpu", tensor_type=torch.FloatTensor):
        """
        prototypes: a p x c stack of p prototype pixels of c channels.
        gwidth2: Gaussian bandwidth squared. Should be small for the count to
            be accurate. But be careful of numerical instability.
        pnorm: specify which norm to use. Could be any positive integer or max.
        """
        super(SoftCountPixels, self).__init__()
        self.prototypes = prototypes
        if gwidth2 <= 0:
            raise ValueError("Bandwidth gwidth2 must be positive. Was {}".format(gwidth2))
        self.gwidth2 = gwidth2
        self.device = device
        self.tensor_type = tensor_type
        self.pnorm = pnorm

    def forward(self, x):
        """
        x: b x c x h x w tensor where b is the batch size, c = #channels.
        """
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")
        pnorm = self.pnorm
        P = self.prototypes
        if P.shape[1] != x.shape[1]:
            raise ValueError(
                "The number of channels must match. prototype has {} channels. But input x has {} channels".format(
                    P.shape[1], x.shape[1]
                )
            )
        npro, chan = P.shape
        b = x.shape[0]
        # Y = output
        Y = torch.empty((b, npro)).to(self.device).type(self.tensor_type)
        for i in range(npro):
            Pi = P[i].view(1, chan, 1, 1)
            # rely on broadcasting for subtraction
            if pnorm == "max":
                sq = torch.abs(x - Pi)
                normsp, _ = torch.max(sq, 1)
            else:
                sq = (x - Pi) ** pnorm
                normsp = torch.sum(sq, 1) ** (1.0 / pnorm)  # batch x h x w
            expo = torch.exp(-normsp / (2.0 * self.gwidth2))
            # print(expo)
            s = torch.sum(expo, (1, 2))
            assert len(s.shape) == 1
            assert s.shape[0] == b
            Y[:, i] = s
        # Ynorm = Y/torch.sum(Y, 1, keepdim=True)
        height, width = x.shape[2], x.shape[3]
        Ynorm = Y / (height * width)
        # Ynorm = Y
        if torch.any(torch.isnan(Ynorm)):
            raise ValueError("Return value contains nan. Perhaps gwidth2 is too small?")

        # import pdb; pdb.set_trace()
        return Ynorm


# end of SoftCountPixels
class MaxColor(nn.Module):
    def __init__(self, device="cpu"):
        super(MaxColor, self).__init__()

        self.maxcolor = nn.AdaptiveMaxPool2d((2, 2))

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")

        # minus statistic
        output = self.maxcolor.forward(x)

        return output.view(output.shape[0], -1)


class StackModule(nn.Module):
    """
    Stack multiple nn.Module's and form a joint output.
    """

    def __init__(self, device="cpu", module_list=[], weights=[]):
        super(StackModule, self).__init__()

        self.nets = module_list  # nn.DataParallel(nn.AdaptiveAvgPool2d(grid_size)).to(device)

        if len(weights) == 0:
            nx = len(module_list)
            weights = [1 / float(nx)] * nx

        assert len(module_list) == len(weights)
        self.weights = weights

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")

        nets = self.nets
        weights = self.weights
        outs = []
        for i, net in enumerate(nets):
            out = net.forward(x)
            out = out.mul(weights[i])
            outs.append(out)

        return torch.cat(outs, dim=1)


class VGG19_face(nn.Module):
    def __init__(self, layers=4, layer_postprocess=None):
        super(VGG19_face, self).__init__()
        self.meta = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "imageSize": [224, 224]}
        self.features_0 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_1 = nn.ReLU()
        self.features_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_3 = nn.ReLU()
        self.features_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_5 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_6 = nn.ReLU()
        self.features_7 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_8 = nn.ReLU()
        self.features_9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_10 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_11 = nn.ReLU()
        self.features_12 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_13 = nn.ReLU()
        self.features_14 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_15 = nn.ReLU()
        self.features_16 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_17 = nn.ReLU()
        self.features_18 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_19 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_20 = nn.ReLU()
        self.features_21 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_22 = nn.ReLU()
        self.features_23 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_24 = nn.ReLU()
        self.features_25 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_26 = nn.ReLU()
        self.features_27 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_28 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_29 = nn.ReLU()
        self.features_30 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_31 = nn.ReLU()
        self.features_32 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_33 = nn.ReLU()
        self.features_34 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_35 = nn.ReLU()
        self.features_36 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.classifier_0 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.classifier_1 = nn.ReLU()
        self.classifier_2 = nn.Dropout(p=0.5)
        self.classifier_3 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier_4 = nn.ReLU()
        self.classifier_5 = nn.Dropout(p=0.5)
        self.classifier_6 = nn.Linear(in_features=4096, out_features=1000, bias=True)

        weights_path = glo.share_path("extractor_models/face/vgg19_pt_mcn.pth")

        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)

        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.layers = layers

        self.layer_postprocess = layer_postprocess

    def forward(self, data):

        assert data.dim() == 4, "Input should have 4 dimensions. Was {}".format(data.dim())

        # This pretrained VGG19 expects input pixels to be in [0,1].
        # See https://pytorch.org/docs/stable/torchvision/models.html
        kmain.pixel_values_check(data, (0, 1), "Input")
        # Normalize according to the instruction on the web page above
        data = self.normalize_module(data)

        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4 = self.features_4(features_3)
        features_5 = self.features_5(features_4)
        features_6 = self.features_6(features_5)
        features_7 = self.features_7(features_6)
        features_8 = self.features_8(features_7)
        features_9 = self.features_9(features_8)
        features_10 = self.features_10(features_9)
        features_11 = self.features_11(features_10)
        features_12 = self.features_12(features_11)
        features_13 = self.features_13(features_12)
        features_14 = self.features_14(features_13)
        features_15 = self.features_15(features_14)
        features_16 = self.features_16(features_15)
        features_17 = self.features_17(features_16)
        features_18 = self.features_18(features_17)
        features_19 = self.features_19(features_18)
        features_20 = self.features_20(features_19)
        features_21 = self.features_21(features_20)
        features_22 = self.features_22(features_21)
        features_23 = self.features_23(features_22)
        features_24 = self.features_24(features_23)
        features_25 = self.features_25(features_24)
        features_26 = self.features_26(features_25)
        features_27 = self.features_27(features_26)
        features_28 = self.features_28(features_27)
        features_29 = self.features_29(features_28)
        features_30 = self.features_30(features_29)
        features_31 = self.features_31(features_30)
        features_32 = self.features_32(features_31)
        features_33 = self.features_33(features_32)
        features_34 = self.features_34(features_33)
        features_35 = self.features_35(features_34)
        features_36 = self.features_36(features_35)

        layers = []
        layers.append(features_0)
        layers.append(features_1)
        layers.append(features_2)
        layers.append(features_3)
        layers.append(features_4)
        layers.append(features_5)
        layers.append(features_6)
        layers.append(features_7)
        layers.append(features_8)
        layers.append(features_9)
        layers.append(features_10)
        layers.append(features_11)
        layers.append(features_12)
        layers.append(features_13)
        layers.append(features_14)
        layers.append(features_15)
        layers.append(features_16)
        layers.append(features_17)
        layers.append(features_18)
        layers.append(features_19)
        layers.append(features_20)
        layers.append(features_21)
        layers.append(features_22)
        layers.append(features_23)
        layers.append(features_24)
        layers.append(features_25)
        layers.append(features_26)
        layers.append(features_27)
        layers.append(features_28)
        layers.append(features_29)
        layers.append(features_30)
        layers.append(features_31)
        layers.append(features_32)
        layers.append(features_33)
        layers.append(features_34)
        layers.append(features_35)
        layers.append(features_36)
        post_process = self.layer_postprocess
        output_layers = [post_process(layers[int(i)]).view(layers[int(i)].shape[0], -1) for i in self.layers]

        classifier_flatten = features_36.view(features_36.size(0), -1)
        classifier_0 = self.classifier_0(classifier_flatten)
        classifier_1 = self.classifier_1(classifier_0)
        classifier_2 = self.classifier_2(classifier_1)
        classifier_3 = self.classifier_3(classifier_2)
        classifier_4 = self.classifier_4(classifier_3)
        classifier_5 = self.classifier_5(classifier_4)
        classifier_6 = self.classifier_6(classifier_5)
        return torch.cat(output_layers, dim=1)


class VGG19(nn.Module):
    """
    Pretrained VGG-19 model features. Layers to use can be specified in the
    constructor. If multiple ones are specified, all the outputs will be
    concatenated.

    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    """

    def __init__(self, layers=4, layer_postprocess=None):
        """
        layers: a list of layers to use to extract features. Outputs will be
            concatenated together.
        layer_postprocess: a post-processing torch.nn.Module to further process
            the result extracted from each layer, before stacking the results
            in the end. y = layer_postprocess(x). Require y.shape[0] =
            x.shape[0] i.e., same minibatch dimension.
        """

        super(VGG19, self).__init__()
        self.layers = layers
        self.model = models.vgg19(pretrained=True).features
        self.layer_postprocess = layer_postprocess
        # Because of large sizes of images we need to downsample images before
        # feeding them to extractors
        # self.downsample = torch.nn.AvgPool2d(3, stride=2)
        for param in self.model.parameters():
            param.requires_grad = False

        # See https://pytorch.org/docs/stable/torchvision/models.html
        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())

        # This pretrained VGG19 expects input pixels to be in [0,1].
        # See https://pytorch.org/docs/stable/torchvision/models.html
        kmain.pixel_values_check(x, (0, 1), "Input")
        # Normalize according to the instruction on the web page above
        x = self.normalize_module(x)

        features = []
        postprocess = self.layer_postprocess
        # if x.size()[-1] ==1024:
        #    x = self.downsample(self.downsample(x))
        for name, layer in enumerate(self.model):
            # name is a non-negative integer.
            x = layer(x)
            if name in self.layers:
                y = postprocess(x) if postprocess is not None else x
                assert y.shape[0] == x.shape[0]
                # flatten
                y_reshaped = y.view(y.shape[0], -1)
                features.append(y_reshaped)
                if len(features) == len(self.layers):
                    # Outputs from all the specified layers are collected.
                    break
        # return features
        return torch.cat(features, dim=1)


# end class VGG19


class GlobalMaxPool(nn.Module):
    """
    A module that takes a 4d tensor X (b x c x h x w) as input and outputs
    a tensor Y of size (b x c). Y is computed by computing the max for each
    input in the batch.
    """

    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be a 4d tensor. Shape was {}".format(x.shape))
        s = x.shape
        xflat = x.view(s[0], s[1], -1)
        y, _ = torch.max(xflat, dim=2)
        assert y.dim() == 2
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == x.shape[1]
        return y


# GlobalMaxPool


class GlobalAvgPool(nn.Module):
    """
    A module that takes a 4d tensor X (b x c x h x w) as input and outputs
    a tensor Y of size (b x c). Y is computed by computing the average for each
    input in the batch.
    """

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be a 4d tensor. Shape was {}".format(x.shape))
        s = x.shape
        xflat = x.view(s[0], s[1], -1)
        y = torch.mean(xflat, dim=2)
        assert y.dim() == 2
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == x.shape[1]
        return y


# GlobalAvgPool


class Identity(nn.Module):
    def __init__(self, flatten=False, slice_dim=None):
        super(Identity, self).__init__()
        self.flatten = flatten
        self.slice = slice_dim

    def forward(self, x):
        if self.slice != None:
            x = x[:, self.slice : (self.slice + 1), :, :]
        return x.view(x.shape[0], -1) if self.flatten else x


class NormalizeChannels2d(nn.Module):
    """
    Normalize (standardize) each channel by a constant i.e.,
    (Tensor[channel_i]-mean[i])/std[i]. Does the same as thing as
    torchvision.transforms.Normalize. But this is a version for tensors with a
    minibatch dimension. Does not modify tensors in place.
    """

    def __init__(self, mean, std):
        """
        mean: list of constants for the means
        std: list of constants used to divide
        """
        super(NormalizeChannels2d, self).__init__()
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        # First dimension of x is the minibatch dimension
        mean = self.mean
        std = self.std
        n_channels = len(mean)
        if n_channels != x.shape[1]:
            raise ValueError(
                "Number of channels of x does not match len of mean vector. x has {} channels. mean length = {}".format(
                    x.shape[1], n_channels
                )
            )
        # This is faster than using broadcasting, don't change without benchmarking
        reshaped_mean = mean.view(1, n_channels, 1, 1)
        reshaped_std = std.view(1, n_channels, 1, 1)
        if torch.cuda.is_available():
            reshaped_mean = reshaped_mean.cuda()
            reshaped_std = reshaped_std.cuda()
        # rely on broadcasting
        standardized = (x - reshaped_mean) / reshaped_std
        return standardized


# end class NormalizeChannels2d


class Flatten(nn.Module):
    """
    A module that flattens even tensor in the minibatch.
    If input tensor has size [n, a, b, c], the this modules returns a tensor
    of size [n, a*b*c] by reshaping.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


# end Flatten


class AlexNet_365(nn.Module):
    """
    Pretrained Alexnet (PLACES-365) model features.
    AlexNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        (1): ReLU(inplace)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (4): ReLU(inplace)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Dropout(p=0.5)
        (1): Linear(in_features=9216, out_features=4096, bias=True)
        (2): ReLU(inplace)
        (3): Dropout(p=0.5)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace)
        (6): Linear(in_features=4096, out_features=365, bias=True)
      )
    )
    """

    def __init__(self):
        """
        layers: a list of layers to use to extract features. Outputs will be
            concatenated together.
        layer_postprocess: a post-processing torch.nn.Module to further process
            the result extracted from each layer, before stacking the results
            in the end. y = layer_postprocess(x). Require y.shape[0] =
            x.shape[0] i.e., same minibatch dimension.
        """

        super(AlexNet_365, self).__init__()
        arch = "alexnet"  #'resnet18'
        # load the pre-trained weights, the weights will be downloaded automatically
        model_file = "%s_places365.pth.tar" % arch
        if not os.access(model_file, os.W_OK):
            weight_url = "http://places2.csail.mit.edu/models_places365/" + model_file
            os.system("wget " + weight_url)

        model = models.__dict__[arch](num_classes=365)

        model = models.__dict__[arch](num_classes=365)
        model = model
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model.features  # Just taking the features

        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")
        x = self.normalize_module(x)
        logit = self.model.forward(x)

        return logit.view(logit.shape[0], -1)


class ResNet18_365(nn.Module):
    """
    Pretrained ResNet-18 (PLACES-365) model features.
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
    )
    """

    def __init__(self):
        super(ResNet18_365, self).__init__()
        arch = "resnet18"
        # load the pre-trained weights, the weights will be downloaded automatically
        model_file = "%s_places365.pth.tar" % arch
        if not os.access(model_file, os.W_OK):
            weight_url = "http://places2.csail.mit.edu/models_places365/" + model_file
            os.system("wget " + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        modules = list(model.children())[:-2]  # Removing the last 2 layers
        model = nn.Sequential(*modules)
        self.model = model.eval()

        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")
        x = self.normalize_module(x)
        logit = self.model.forward(x)

        return logit.view(logit.shape[0], -1)


class ResNet50_365(nn.Module):
    """
    Pretrained ResNet-50 (PLACES-365) model features.
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (5): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (6): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (7): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
    )
    """

    def __init__(self, n_remove_last_layers=2, layer_postprocess=None):
        """
        n_remove_last_layers: number of layers to remove (count from the last layer).
            If 0, don't remove.
        """

        super(ResNet50_365, self).__init__()
        assert n_remove_last_layers >= 0
        arch = "resnet50"
        # load the pre-trained weights, the weights will be downloaded automatically
        model_file = "%s_places365.pth.tar" % arch
        if not os.access(model_file, os.W_OK):
            weight_url = "http://places2.csail.mit.edu/models_places365/" + model_file
            # TODO: wget depends on the OS. Fix it to make it work on Windows as well.
            os.system("wget " + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        modules = list(model.children())
        if n_remove_last_layers > 0:
            modules = modules[:-n_remove_last_layers]  # remove some layers from the last
        model = nn.Sequential(*modules)
        self.model = model.eval()

        self.layer_postprocess = layer_postprocess

        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        kmain.pixel_values_check(x, (0, 1), "Input")
        x = self.normalize_module(x)
        logit = self.model.forward(x)
        logit = self.layer_postprocess(logit)

        return logit.view(logit.shape[0], -1)


class ResNet18_365Layer(nn.Module):
    """
    Training MultiLabel Classifier on the Pretrained ResNet-18 (PLACES-365) model features.
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
    )
    """

    def __init__(self):
        """
        layers: a list of layers to use to extract features. Outputs will be
            concatenated together.
        layer_postprocess: a post-processing torch.nn.Module to further process
            the result extracted from each layer, before stacking the results
            in the end. y = layer_postprocess(x). Require y.shape[0] =
            x.shape[0] i.e., same minibatch dimension.
        """

        super(ResNet18_365Layer, self).__init__()
        arch = "resnet18"
        # load the pre-trained weights, the weights will be downloaded automatically
        model_file = "%s_places365.pth.tar" % arch
        if not os.access(model_file, os.W_OK):
            weight_url = "http://places2.csail.mit.edu/models_places365/" + model_file
            os.system("wget " + weight_url)

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        # self.model.eval()

        self.normalize_module = NormalizeChannels2d(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, "Input should have 4 dimensions. Was {}".format(x.dim())
        # kmain.pixel_values_check(x, (0,1), 'Input')
        x = self.normalize_module(x)
        # x = self.model(x)
        return self.model.forward(x)


# class PTImgFeatureExtractor(torch.nn.Module):
#     """
#     An abstract class for a feature extractor for images. Pytorch version.
#     A subclass of torch.nn.Module. forward() function does feature extraction.
#     """

#     def extract(self, imgs):
#         """
#         imgs: A stack of images such that imgs[i,...] is one image.
#             Each image can have more than one channel.

#         Return a Pytorch tensor T such that T.shape[0] == imgs.shape[0], and
#             T[i,:,...:] is the feature vector (or tensor) of the ith image.
#         """
#         return self.forward(imgs)

#     def __call__(self, imgs):
#         return self.extract(imgs)

# # end PTImgFeatureExtractor
