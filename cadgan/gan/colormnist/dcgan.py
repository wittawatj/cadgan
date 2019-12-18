from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, bs=64):
        super(Generator, self).__init__()
        self.bs = bs
        self.deconv1 = nn.ConvTranspose2d(100, bs*8, 4) #, stride = 1, padding = 0) # 100 in, 1024 outchannels
        self.bn1 = nn.BatchNorm2d(bs*8) #1024 channels
        self.deconv2 = nn.ConvTranspose2d(bs*8, bs*4, 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(bs*4)
        self.deconv3 = nn.ConvTranspose2d(bs*4, bs*2, 4, stride = 2, padding = 2)
        self.bn3 = nn.BatchNorm2d(bs*2)
        self.deconv4 = nn.ConvTranspose2d(bs*2, 3, 4, stride = 2, padding = 1)
        #self.bn4 = nn.BatchNorm2d(bs)
        #self.deconv5 = nn.ConvTranspose2d(bs, 3, 4, stride=2, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        input = input.view(-1,100,1,1)
        out = self.relu(self.bn1(self.deconv1(input)))
        out = self.relu(self.bn2(self.deconv2(out)))
        out = self.relu(self.bn3(self.deconv3(out)))
        out = self.sigmoid(self.deconv4(out))
        return out
