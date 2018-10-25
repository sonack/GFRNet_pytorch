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


class ToTensor(object):
     def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        sample['img_l'], sample['img_r'] = to_tensor(sample['img_l']), to_tensor(sample['img_r'])
        return sample

