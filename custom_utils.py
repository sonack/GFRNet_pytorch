# from __future__ import print_function, division
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
# import dataset
import models

from opts import opt
from os import path
from torchvision.utils import save_image
import torch.nn.functional as F
from termcolor import colored


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def debug_print(msg, color='red'):
    print (colored(msg, color))

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


def file_suffix(filename):
    return path.splitext(filename)[-1]


# [1,256] --> [-1, 1]
def normalize(num):
    return (num - 1) / 255 * 2 - 1

# [-1, 1] --> [1,256]
def denormalize(num):
    return (num + 1) / 2 * 255 + 1

# clamp to 0~255
def valid(num):
    return max(min(num, 255), 0)

def save_result_imgs(img_path, triplet):
    guide, gt, warp_guide = triplet
    batch_size = guide.size()[0]
    img_basenames = list(map(path.basename, img_path))
    img_suffixes = [file_suffix(name) for name in img_basenames]
    img_ids = [name.split('_')[0] for name in img_basenames]
    img_filenames = [path.join(opt.save_imgs_dir, img_id + img_suffix) for img_id, img_suffix in zip(img_ids, img_suffixes)]

    # print (img_filenames)
    for batch_id in range(batch_size):
        triplet_tensor = torch.cat([guide[batch_id, ... ], gt[batch_id, ... ], warp_guide[batch_id, ... ]], 0).view(3, 3, opt.img_size, opt.img_size)
        print ('save image %s ...' % img_filenames[batch_id])
        save_image(triplet_tensor, img_filenames[batch_id], 3)


# def save_test_imgs(img_path, couple):
#     gt, restored_img = couple
#     batch_size = gt.size()[0]
#     img_basenames = list(map(path.basename, img_path))
#     img_suffixes = [file_suffix(name) for name in img_basenames]
#     img_ids = [name.split('_')[0] for name in img_basenames]
#     img_filenames = [path.join(opt.save_imgs_dir, img_id + img_suffix) for img_id, img_suffix in zip(img_ids, img_suffixes)]

#     # print (img_filenames)
#     for batch_id in range(batch_size):
#         couple_tensor = torch.cat([gt[batch_id, ... ], restored_img[batch_id, ... ]], 0).view(2, 3, opt.img_size, opt.img_size)
#         print ('save image %s ...' % img_filenames[batch_id])
#         save_image(couple_tensor, img_filenames[batch_id], 2)
    # pdb.set_trace()

# crop the local face region of a batch of imgs, and resize it to opt.img_size squared batch
def crop_face_region_batch(imgs, face_regions):
    crop_imgs = torch.empty_like(imgs)
    batch_size = imgs.size(0)
    for batch_id in range(batch_size):
        x1 = face_regions[0][0][batch_id]
        y1 = face_regions[0][1][batch_id]
        x2 = face_regions[1][0][batch_id]
        y2 = face_regions[1][1][batch_id]
        # print ('left top: (%d, %d), right bottom: (%d, %d)' % (x1, y1, x2, y2))
        tmp = imgs[batch_id:batch_id+1,:,y1:y2+1,x1:x2+1]
        # print ("tmp shape", tmp.shape)
        # pdb.set_trace()
        # crop_imgs = torch.empty_like(tmp)
        crop_imgs[batch_id] = F.interpolate(tmp, size=opt.img_size, mode='bilinear', align_corners=True)[0]
        # crop_imgs[batch_id] = tmp[0]
    return crop_imgs


def crop_part_region_batch(imgs, part_pos):
    batch_size = imgs.size(0)
    part_tensor_size = (batch_size, 3, opt.part_size, opt.part_size)

    device = imgs.device
    L = torch.empty(*part_tensor_size, device=device)
    R = torch.empty(*part_tensor_size, device=device)
    N = torch.empty(*part_tensor_size, device=device)
    M = torch.empty(*part_tensor_size, device=device)
    parts = [L, R, N, M]
    
    for part_id in range(4):
        for batch_id in range(batch_size):
            mid_x = part_pos[batch_id, part_id, 0]
            mid_y = part_pos[batch_id, part_id, 1]
            half_len = part_pos[batch_id, part_id, 2] / 2
        #     print ('center=(%d, %d)  len=(%d)' % (mid_x, mid_y, half_len))
            x1 =  max(mid_x - half_len, 0)
            y1 = max(mid_y - half_len, 0)
            x2 = min(mid_x + half_len, opt.img_size - 1)
            y2 = min(mid_y + half_len, opt.img_size - 1)
            # print ('left top: (%d, %d), right bottom: (%d, %d)' % (x1, y1, x2, y2))
            tmp = imgs[batch_id:batch_id+1,:,y1:y2+1,x1:x2+1]
            parts[part_id][batch_id] = F.interpolate(tmp, size=opt.part_size, mode='bilinear', align_corners=True)[0]
    return parts

def make_face_region_mask(imgs, mask_weight, face_regions):
    masks = torch.ones_like(imgs)
    # masks = torch.zeros_like(imgs)

    batch_size = imgs.size(0)
    for batch_id in range(batch_size):
        x1 = face_regions[0][0][batch_id]
        y1 = face_regions[0][1][batch_id]
        x2 = face_regions[1][0][batch_id]
        y2 = face_regions[1][1][batch_id]
        masks[batch_id,:,y1:y2+1,x1:x2+1] = mask_weight
    return masks


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




