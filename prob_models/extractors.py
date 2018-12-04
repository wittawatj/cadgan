#coding:utf8
import torch
import torch.nn as nn
from torchvision.models import vgg19
from collections import namedtuple
import torch.nn.init as init
from math import sqrt
import torchvision.models as models
import os

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
            raise ValueError('mean and std must have the same length')
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        # First dimension of x is the minibatch dimension
        mean = self.mean
        std = self.std
        n_channels = len(mean)
        if n_channels != x.shape[1]:
            raise ValueError('Number of channels of x does not match len of mean vector. x has {} channels. mean length = {}'.format(x.shape[1], n_channels))
        # This is faster than using broadcasting, don't change without benchmarking
        reshaped_mean = mean.view(1, n_channels, 1, 1)
        reshaped_std = std.view(1, n_channels, 1, 1)
        if torch.cuda.is_available():
            reshaped_mean = reshaped_mean.cuda()
            reshaped_std = reshaped_std.cuda()
        # rely on broadcasting
        standardized = (x - reshaped_mean)/reshaped_std
        return standardized

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
    def __init__(self, num_classes):
        """
        layers: a list of layers to use to extract features. Outputs will be
            concatenated together.
        layer_postprocess: a post-processing torch.nn.Module to further process
            the result extracted from each layer, before stacking the results
            in the end. y = layer_postprocess(x). Require y.shape[0] =
            x.shape[0] i.e., same minibatch dimension.
        """

        super(ResNet18_365, self).__init__()
        arch = 'resnet18'
        # load the pre-trained weights, the weights will be downloaded automatically
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        modules=list(model.children())[:-2] # Removing the last 2 layers
        self.model = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)
        #self.model = model.eval()
        self.num_classes = num_classes

        self.normalize_module = NormalizeChannels2d(
                mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    def forward(self, x):
        assert x.dim() == 4, 'Input should have 4 dimensions. Was {}'.format(x.dim())
        #kmain.pixel_values_check(x, (0,1), 'Input')
        x = self.normalize_module(x)
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
