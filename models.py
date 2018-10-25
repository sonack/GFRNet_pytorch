from __future__ import print_function, division
import os
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import namedtuple
import pdb
from stn_module import STN
from opts import opt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


warpNet_encoder = nn.Sequential(
    nn.Conv2d(6, opt.ngf, kernel_size=4, stride=2, padding=1),
    # 128
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf, opt.ngf * 2, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 2),
    # 64
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 2, opt.ngf * 4, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 4),
    # 32
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 4, opt.ngf * 8, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 8),
    # 16
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 8, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 8
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 4 
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 2
    nn.LeakyReLU(0.2),
    nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
    # nn.BatchNorm2d(opt.ngf * 16),
    # 1
)

warpNet_decoder = nn.Sequential(
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 2
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 4
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 16),
    # 8
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 8),
    # 16
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 4),
    # 32
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf * 2),
    # 64
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1),
    nn.BatchNorm2d(opt.ngf),
    # 128
    nn.ReLU(),
    nn.ConvTranspose2d(opt.ngf, opt.output_nc, 4, 2, 1),
    nn.Tanh(),
    # grid [-1,1]
)

# warpNet output flow field
warpNet = nn.Sequential(
    warpNet_encoder,
    warpNet_decoder
)

class GFRNet_warpnet(nn.Module):

    def __init__(self):
        super(GFRNet_warpnet, self).__init__()
        self.warpNet = warpNet
        self.stn = STN()
        
    def forward(self, gt, guide):
        pair = torch.cat([gt, guide], 1)  # C = 6
        grid = self.warpNet(pair) # NCHW
        grid_NHWC = grid.permute(0,2,3,1)
        warp_guide = self.stn(guide, grid_NHWC)
        return warp_guide, grid



