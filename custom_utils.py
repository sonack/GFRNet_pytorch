from __future__ import print_function, division
import os
from os import path
import torch
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

import argparse
import random
import dataset
import models

from opts import opt

def create_orig_xy_map():
    x = torch.linspace(-1, 1, opt.img_size)
    y = torch.linspace(-1, 1, opt.img_size)
    grid_y, grid_x = torch.meshgrid([x, y])
    grid_x = grid_x.view(1, 1, opt.img_size, opt.img_size)
    grid_y = grid_y.view(1, 1, opt.img_size, opt.img_size)
    orig_xy_map = torch.cat([grid_x, grid_y], 1) # channel stack
    # print (orig_xy_map)
    # pdb.set_trace()
    return orig_xy_map
