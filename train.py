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


import custom_transforms
from criterions import MaskedMSELoss, TVLoss, MultiSymLoss
from tensorboardX import SummaryWriter
import time

from custom_utils import create_orig_xy_map

from opts import opt
from pprint import pprint

pprint (vars(opt))

# random seed
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)


# device use
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")

print ('Use device: %s' % device)

# datasets
tsfm = transforms.Compose([
    custom_transforms.ToTensor()
])


face_dataset = dataset.FaceDataset(opt.subset, opt.landmark_dir, opt.sym_dir, opt.img_dir, tsfm)

face_dataset_dataloader = DataLoader(face_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)

# model
GFRNet_warpnet = models.GFRNet_warpnet()
GFRNet_warpnet.to(device)

# print (GFRNet_warpnet)

# optimizer
optimizer = torch.optim.Adam(GFRNet_warpnet.parameters(), lr=opt.lr)

# load ckpt
last_epoch = -1

if opt.load_checkpoint:
    print ()
    print ('=' * 20)
    print ('load checkpoint from %s', opt.load_checkpoint)
    ckpt = torch.load(opt.load_checkpoint)
    GFRNet_warpnet.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    last_epoch = ckpt['epoch']
    last_tv_loss = ckpt['tv_loss']
    last_point_loss = ckpt['point_loss']
    print ('cont training from epoch %2d' % (last_epoch + 1))
    print ('last tv loss is %f' % last_tv_loss)
    print ('last point loss is %f' % last_point_loss)
    print ('=' * 20)
    print ()



# criterions

point_crit = MaskedMSELoss()
tv_crit = TVLoss()
sym_crit = MultiSymLoss(opt.sym_loss_C_start, opt.sym_loss_C_step, opt.sym_loss_C_end)


# others
orig_xy_map = create_orig_xy_map().to(device)
writer = SummaryWriter(opt.exp_name)

i_batch_tot = 0

# set mode
GFRNet_warpnet.train()

for epoch in range(last_epoch + 1, opt.num_epochs):

    avg_tv_loss = 0
    avg_point_loss = 0
    avg_sym_loss = 0

    for i_batch, sample_batched in enumerate(face_dataset_dataloader):
        # just for debug
        # if i_batch < 810:
        #     continue
        # print ('I_batch:', i_batch)
        
        # if i_batch == 2:
        #     break
        
        # print(i_batch, sample_batched['img_l'].size(),
        #       sample_batched['lm_l'].size())

        # sample prepare
        gt = sample_batched['img_l'].to(device)
        guide = sample_batched['img_r'].to(device)
        
        lm_mask = sample_batched['lm_mask'].to(device)
        lm_gt = sample_batched['lm_gt'].to(device)

        sym_r = sample_batched['sym_r'].to(device)

        # pdb.set_trace()


        # forward
        warp_guide, grid = GFRNet_warpnet(gt, guide)

        GFRNet_warpnet.zero_grad()
        
        # calc loss
        point_loss = point_crit(grid, lm_gt, lm_mask)
        tv_loss = tv_crit(grid - orig_xy_map)
        sym_loss = sym_crit(grid, sym_r)

        total_loss = opt.point_loss_weight * point_loss + opt.tv_loss_weight * tv_loss + opt.sym_loss_weight * sym_loss

        if not opt.just_look:
            # print ('backward & update params!')
            total_loss.backward()
            optimizer.step()

        # tensorboardX display
        if i_batch % opt.disp_freq == 0:
            # writer.add_image('guide', guide[:opt.disp_img_cnt], i_batch_tot)
            # writer.add_image('groundtruth', gt[:opt.disp_img_cnt], i_batch_tot)
            # writer.add_image('warp guide', warp_guide[:opt.disp_img_cnt], i_batch_tot)

            writer.add_image('guide-gt-warp', torch.cat([guide[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], warp_guide[:opt.disp_img_cnt]], 2), i_batch_tot)

            writer.add_scalar('loss/point_loss', point_loss.item(), i_batch_tot)
            writer.add_scalar('loss/tv_loss', tv_loss.item(), i_batch_tot)
            writer.add_scalar('loss/sym_loss', sym_loss.item(), i_batch_tot)
            writer.add_scalar('loss/tot_loss', total_loss.item(), i_batch_tot)
            

        # pdb.set_trace()
        # print loss
        if i_batch % opt.print_freq == 0:
            print ('Time: %s [(%d/%d) ; (%d/%d)] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f' % 
                    (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i_batch, len(face_dataset_dataloader), \
                    epoch, opt.num_epochs, total_loss.item(), opt.point_loss_weight * point_loss.item(), \
                    opt.tv_loss_weight * tv_loss.item(), opt.sym_loss_weight * sym_loss.item()))
        
        avg_tv_loss += (opt.tv_loss_weight * tv_loss.item()) / len(face_dataset_dataloader)
        avg_point_loss += (opt.point_loss_weight * point_loss.item()) / len(face_dataset_dataloader)
        avg_sym_loss += (opt.sym_loss_weight * sym_loss.item()) / len(face_dataset_dataloader)

        i_batch_tot += 1
    
    # epoch level loss info
    print ()
    print ('=' * 20)
    print ('Time: %s Epoch [%d/%d] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f' % 
                    (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, opt.num_epochs, avg_point_loss + avg_tv_loss + avg_sym_loss, avg_point_loss, avg_tv_loss, avg_sym_loss))
    print ('=' * 20)
    print ()
    
    # save model
    if (not opt.just_look) and ((epoch + 1) % opt.save_epoch_freq == 0):
        
        checkpoint_file = path.join(opt.checkpoint_dir, 'checkpoint_%02d.pt' % (epoch+1))
        # print ('epoch is', epoch, 'opt.save_epoch_freq is', opt.save_epoch_freq)
        print ('save model to %s ...' % checkpoint_file)
        torch.save({
            'epoch': epoch,
            'model_state_dict': GFRNet_warpnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tv_loss': avg_tv_loss,
            'point_loss': avg_point_loss,
            'sym_loss': avg_sym_loss,
            }, checkpoint_file)
        

writer.close()



def test():
    create_orig_xy_map()  

if __name__ == '__main__':
    # test()
    pass
