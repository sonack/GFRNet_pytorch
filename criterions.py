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
import torch.nn as nn

from netVggs.netVgg_conv3 import netVgg_conv3
from netVggs.netVgg_conv4 import netVgg_conv4

from opts import opt

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction=None):
        super(MaskedMSELoss, self).__init__()
        if reduction:
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            self.criterion = nn.MSELoss()


    def forward(self, input, target, mask):
        # pdb.set_trace()
        self.loss = self.criterion(input * mask, target)
        return self.loss


# https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class SymLoss(nn.Module):
    def __init__(self, C):
        super(SymLoss, self).__init__()
        self.C = C
    
    def forward(self, grid, sym_axis):
        # grid (N,2,H,W)
        batch_size = grid.size()[0]
        h = grid.size()[2]
        w = grid.size()[3]
        # h change to h - C    
        sym_x = sym_axis[:,0].view(batch_size, 1, 1)
        # ny --> -ny
        sym_y = sym_axis[:,1].view(batch_size, 1, 1)

        
        # print ('hi~')
        
        
        delta_grid_x = grid[:,0,:h-self.C,:] - grid[:,0,self.C:,:]
        delta_grid_y = grid[:,1,:h-self.C,:] - grid[:,1,self.C:,:]

        # print ('delta_grid_x.size:')
        # print (delta_grid_x.size())

        # pdb.set_trace()
        sym_loss = torch.pow(delta_grid_x * sym_y + delta_grid_y * sym_x, 2).sum()
        return sym_loss / ((h - self.C) * w * batch_size)


class MultiSymLoss(nn.Module):
    def __init__(self, C_start, C_step, C_end):
        super(MultiSymLoss, self).__init__()
        print ('MultiSymLoss C list is %s' % list(range(C_start, C_end+1, C_step)))
        self.sym_losses = nn.ModuleList([SymLoss(C) for C in range(C_start, C_end+1, C_step)])

    
    def forward(self, grid, sym_axis):
        loss = 0
        for i, sym_loss in enumerate(self.sym_losses):
            loss += sym_loss(grid, sym_axis)
        return loss

class VggFaceLoss(nn.Module):
    def __init__(self, device, ver = 3):
        super(VggFaceLoss, self).__init__()
        self.netVgg = netVgg_conv3 if ver == 3 else netVgg_conv4
        self.netVgg.load_state_dict(torch.load('./netVggs/netVgg_conv%d.pth' % ver))
        self.criterion = nn.MSELoss()
        self.RGB_mean = torch.tensor([129.1863,104.7624,93.5940], device=device).view(1, 3, 1, 1)
        # pdb.set_trace()
        for param in self.netVgg.parameters():
            param.requires_grad = False
    
    def forward(self, restored, gt):
        if opt.minusone_to_one:  # [-1, 1]
            zero_to_one_restored = restored * 0.5 + 0.5
            zero_to_one_gt = gt * 0.5 + 0.5
        else:
            zero_to_one_restored = restored
            zero_to_one_gt = gt
        restored_vgg = zero_to_one_restored * 255 - self.RGB_mean
        gt_vgg = zero_to_one_gt * 255  - self.RGB_mean
        # pdb.set_trace()
        gt_feat = self.netVgg(gt_vgg)
        res_feat = self.netVgg(restored_vgg)
        loss = self.criterion(res_feat, gt_feat)
        return loss
    




