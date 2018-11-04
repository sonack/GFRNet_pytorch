from __future__ import print_function, division
import os
from os import path
import torch
import torch.nn as nn
from skimage import io, transform
import numpy as np
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
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
from criterions import MaskedMSELoss, TVLoss, MultiSymLoss, VggFaceLoss
from tensorboardX import SummaryWriter
import time

from custom_utils import create_orig_xy_map, save_result_imgs, crop_face_region_batch, crop_part_region_batch, make_face_region_mask, debug_print, pretty, weight_init, norm_to_01

from opts import opt
from pprint import pprint
import json
from torch.optim import lr_scheduler
from termcolor import colored
import json

configs = json.dumps(vars(opt), indent=2)
print (colored(configs, 'green'))
# print (colored(vars(opt), 'green'))
# pretty (vars(opt))

opts_json_path = path.join(opt.checkpoint_dir, 'opts.json')

if not opt.just_look:
    with open(opts_json_path, 'w') as f:
        print ('save opts to %s' % opts_json_path)
        json.dump(vars(opt), f)

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

tsfms = [
    custom_transforms.DegradationModel(),
    custom_transforms.ToTensor()
]
if opt.minusone_to_one:
    tsfms.append(custom_transforms.NormalizeToMinusOneToOne())
tsfm = transforms.Compose(tsfms)


face_dataset = dataset.FaceDataset(opt.subset, opt.landmark_dir, opt.sym_dir, opt.img_dir, opt.flip_prob, tsfm)

face_dataset_dataloader = DataLoader(face_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)

# model
# GFRNet_warpnet = models.GFRNet_warpnet()
# GFRNet_warpnet.to(device)
if opt.only_train_warpnet:
    GFRNet_warpnet = models.GFRNet_warpnet()
    GFRNet_warpnet.to(device)
else:
    GFRNet_G = models.GFRNet_generator()
    GFRNet_G.to(device)
    if opt.use_custom_init:
        GFRNet_G.apply(weight_init)
    if opt.use_gan_loss:
        GFRNet_globalD = models.GFRNet_globalDiscriminator(opt.input_nc_img*2+opt.output_nc_img) # cond  [wg, g, gt/res]
        GFRNet_globalD.to(device)
        if opt.use_custom_init:
            GFRNet_globalD.apply(weight_init)
        if opt.use_part_gan:
            GFRNet_partD_L = models.GFRNet_partDiscriminator(opt.input_nc_img + opt.output_nc_img) # cond  [part wg, part gt/res] in C dim,  left is warp guide, right is gt/res
            GFRNet_partD_R = models.GFRNet_partDiscriminator(opt.input_nc_img + opt.output_nc_img)
            GFRNet_partD_N = models.GFRNet_partDiscriminator(opt.input_nc_img + opt.output_nc_img)
            GFRNet_partD_M = models.GFRNet_partDiscriminator(opt.input_nc_img + opt.output_nc_img)
            GFRNet_partDs = [GFRNet_partD_L, GFRNet_partD_R, GFRNet_partD_N, GFRNet_partD_M]

            for GFRNet_partD in GFRNet_partDs:
                GFRNet_partD.to(device)
                if opt.use_custom_init:
                    GFRNet_partD.apply(weight_init)
        else:
            GFRNet_localD = models.GFRNet_localDiscriminator(opt.input_nc_img) # uncond
            GFRNet_localD.to(device)
            if opt.use_custom_init:
                GFRNet_localD.apply(weight_init)
       
        
        
        



# pdb.set_trace()

# print (GFRNet_warpnet)

# optimizer
betas = (opt.beta1, 0.999)
# optimizer = torch.optim.Adam(GFRNet_warpnet.parameters(), lr=opt.lr)
if opt.only_train_warpnet:
    optimizer = torch.optim.Adam(GFRNet_warpnet.parameters(), lr=opt.lr, betas=betas)
else:
    # optimizerWarpNet = torch.optim.Adam(GFRNet_G.warpNet.parameters(), lr=opt.lr)
    # optimizerRecNet = torch.optim.Adam(GFRNet_G.recNet.parameters(), lr=opt.lr)
    # optimizerG = torch.optim.Adam(GFRNet_G.parameters(), lr=opt.lr)
    optimizerG = torch.optim.Adam(
        [
            { 'params': GFRNet_G.warpNet.parameters(), 'lr': opt.lr * 0.001 },
            { 'params': GFRNet_G.recNet.parameters() }
        ],
        lr=opt.lr,
        betas=betas
    )

    if opt.use_gan_loss:
        optimizerGlobalD = torch.optim.Adam(GFRNet_globalD.parameters(), lr=opt.lr, betas=betas)
        if opt.use_part_gan:
            optimizerPartD_L = torch.optim.Adam(GFRNet_partD_L.parameters(), lr=opt.lr, betas=betas)
            optimizerPartD_R = torch.optim.Adam(GFRNet_partD_R.parameters(), lr=opt.lr, betas=betas)
            optimizerPartD_N = torch.optim.Adam(GFRNet_partD_N.parameters(), lr=opt.lr, betas=betas)
            optimizerPartD_M = torch.optim.Adam(GFRNet_partD_M.parameters(), lr=opt.lr, betas=betas)
            partD_optims = [optimizerPartD_L, optimizerPartD_R, optimizerPartD_N, optimizerPartD_M]
        else:
            optimizerLocalD = torch.optim.Adam(GFRNet_localD.parameters(), lr=opt.lr, betas=betas)


# load ckpt
last_epoch = -1


if opt.load_checkpoint:
    if opt.only_train_warpnet:
        print ()
        print ('=' * opt.sep_width)
        print ('load checkpoint from %s' % opt.load_checkpoint)
        ckpt = torch.load(opt.load_checkpoint)
        GFRNet_warpnet.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        last_epoch = ckpt['epoch']
        last_tv_loss = ckpt['tv_loss']
        last_point_loss = ckpt['point_loss']
        print ('cont training from epoch %2d' % (last_epoch + 1))
        print ('last tv loss is %f' % last_tv_loss)
        print ('last point loss is %f' % last_point_loss)
        print ('=' * opt.sep_width)
        print ()
    else:
        print ()
        print ('=' * opt.sep_width)
        print ('load checkpoint from %s' % opt.load_checkpoint)
        ckpt = torch.load(opt.load_checkpoint)
        use_gan_loss = ckpt.get('use_gan_loss', True)
        assert opt.use_gan_loss == use_gan_loss, 'The loaded ckpt use_gan_loss=%d , while current config use_gan_loss=%d' % (use_gan_loss, opt.use_gan_loss)
        # load models & optimizers
        GFRNet_G.load_state_dict(ckpt['G_state_dict'])
        optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])

        if opt.use_gan_loss:
            # snk
            GFRNet_globalD.load_state_dict(ckpt['globalD_state_dict'])
            optimizerGlobalD.load_state_dict(ckpt['optimizerGlobalD_state_dict'])
            if opt.use_part_gan:
                for idx, part in enumerate(['L', 'R', 'N', 'M']):
                    GFRNet_partDs[idx].load_state_dict(ckpt['partD_%s_state_dict' % part])
                    partD_optims[idx].load_state_dict(ckpt['optimizerPartD_%s_state_dict' % part])
            else:
                GFRNet_localD.load_state_dict(ckpt['localD_state_dict'])
                optimizerLocalD.load_state_dict(ckpt['optimizerLocalD_state_dict'])

        last_epoch = ckpt['epoch']
        # losses [ignore temporarily]
        # last_tv_loss = ckpt['tv_loss']
        # last_point_loss = ckpt['point_loss']
        print ('cont training from epoch %2d' % (last_epoch + 1))
        # print ('last tv loss is %f' % last_tv_loss)
        # print ('last point loss is %f' % last_point_loss)
        print ('=' * opt.sep_width)
        print ()
        # print (GFRNet_G.warpNet.warpNet[0][0].weight.norm())
        # pdb.set_trace()
else:
    if opt.load_warpnet_ckpt and opt.load_warpnet_ckpt != 'None':
        print ()
        print ('=' * opt.sep_width)
        print ('load pretrained warpNet checkpoint from %s' % opt.load_warpnet_ckpt)
        ckpt = torch.load(opt.load_warpnet_ckpt)
        GFRNet_G.warpNet.load_state_dict(ckpt['model_state_dict'])
        print ('=' * opt.sep_width)
        print ()
        # print (GFRNet_G.warpNet.warpNet[0][0].weight.norm())
        # pdb.set_trace()


# criterions

point_crit = MaskedMSELoss()
tv_crit = TVLoss()
sym_crit = MultiSymLoss(opt.sym_loss_C_start, opt.sym_loss_C_step, opt.sym_loss_C_end)

if not opt.only_train_warpnet:
    if opt.use_gan_loss:
        if opt.use_lsgan:
            D_crit = nn.MSELoss
        else:
            D_crit = nn.BCELoss
    
        globalD_crit = D_crit()        
        if opt.use_part_gan:
            partD_L_crit = D_crit()
            partD_R_crit = D_crit()
            partD_N_crit = D_crit()
            partD_M_crit = D_crit()
            partD_crits = [partD_L_crit, partD_R_crit, partD_N_crit, partD_M_crit]
        else:
            localD_crit = D_crit()
        


    # rec_mse_crit = nn.MSELoss(reduction='sum') # eq to size_average = False
    rec_face_mse_crit_show = nn.MSELoss(reduction='sum')
    # rec_weighted_mse_crit = MaskedMSELoss(reduction='sum')
    rec_mse_crit = MaskedMSELoss(reduction='sum')
    rec_face_mse_crit = nn.MSELoss(reduction='sum')
    rec_perp_crit = VggFaceLoss(device, 3)
    rec_perp_crit.to(device)
    # rec_perp_crit = nn.MSELoss() # vggface

# lr scheduler


if not opt.only_train_warpnet:
    print ('last epoch:', last_epoch)
    schedulerG = lr_scheduler.StepLR(optimizerG, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_rate, last_epoch=last_epoch)
    schedulerGlobalD = lr_scheduler.StepLR(optimizerGlobalD, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_rate, last_epoch=last_epoch)
    all_schedulers = [schedulerG, schedulerGlobalD]
    if opt.use_part_gan:
        for partD_optim in partD_optims:
            all_schedulers.append(lr_scheduler.StepLR(partD_optim, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_rate, last_epoch=last_epoch))
    else:
        schedulerLocalD = lr_scheduler.StepLR(optimizerLocalD, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_rate, last_epoch=last_epoch)
        all_schedulers.append(schedulerLocalD)

# schedulers = [schedulerWarpNet, schedulerRecNet, schedulerGlobalD, schedulerLocalD]

# others
orig_xy_map = create_orig_xy_map().to(device)
if not opt.save_imgs:
    writer = SummaryWriter(opt.exp_name)
    writer.add_text('Configs', configs, 0)

real_label_val = 1
fake_label_val = 0



i_batch_tot = 0

# set mode
if opt.only_train_warpnet:
    GFRNet_warpnet.train()
else:
    GFRNet_G.train()
    if opt.use_gan_loss:
        GFRNet_globalD.train()
        if opt.use_part_gan:
            for GFRNet_partD in GFRNet_partDs:
                GFRNet_partD.train()
        else:
            GFRNet_localD.train()

def print_inter_grad(msg):
    def func(x):
        print (msg)
        print (x.norm().item())
        # x.data = x.data * 1e-10
    return func


for epoch in range(last_epoch + 1, opt.num_epochs):

    avg_tv_loss = 0
    avg_point_loss = 0
    avg_sym_loss = 0
    avg_globalD_loss_for_G = 0
    avg_localD_loss_for_G = 0
    avg_globalD_loss_for_D = 0
    avg_localD_loss_for_D = 0
    avg_rec_mse_loss = 0
    avg_rec_face_mse_loss = 0
    avg_rec_perp_loss = 0

    batch_num_per_epoch = len(face_dataset_dataloader)
    
    for scheduler in all_schedulers:
        scheduler.step()
    
    # param_groups[0]['lr'] is 0.001 x cur_lr
    cur_lr = optimizerG.param_groups[1]['lr']
    # pdb.set_trace()
    writer.add_scalar('lr', cur_lr, epoch)
    # print ('cur_lr', cur_lr)

    for i_batch, sample_batched in enumerate(face_dataset_dataloader):
        # just for debug
        # print ('i_batch:', i_batch)
        # if i_batch < 810:
        #     continue
        # print ('I_batch:', i_batch)
        
        # pdb.set_trace()
        # if i_batch == 5:
        #     break
        
        # print(i_batch, sample_batched['img_l'].size(),
        #       sample_batched['lm_l'].size())

        # sample prepare


       
        blur = sample_batched['img_l'].to(device)
        guide = sample_batched['img_r'].to(device)
        gt = sample_batched['gt'].to(device)
        
        lm_mask = sample_batched['lm_mask'].to(device)
        lm_gt = sample_batched['lm_gt'].to(device)

        sym_r = sample_batched['sym_r'].to(device)

       
        

        if not opt.only_train_warpnet:

            if opt.use_gan_loss:
                real_batch_size = blur.size(0)
                real_label_val = random.randint(7, 13) / 10
                fake_label_val = random.randint(0, 4) / 10
                real_label = torch.full((real_batch_size, 1, 1, 1), real_label_val, device=device)
                fake_label = torch.full((real_batch_size, 1, 1, 1), fake_label_val, device=device)

            face_region = sample_batched['face_region_calc']
            part_pos = sample_batched['part_pos']
            # pdb.set_trace()
            localD_input_real_single = crop_face_region_batch(gt, face_region)
            face_region_blur = crop_face_region_batch(blur, face_region)
            part_region_blur = crop_part_region_batch(blur, part_pos)
            # print ('face region inspect ... ')
  
            # L_blur, R_blur, N_blur, M_blur = crop_part_region_batch(gt, part_pos)
            # writer.add_image('test/local', torch.cat([localD_input_real_single[:opt.disp_img_cnt], face_region_blur[:opt.disp_img_cnt]], 2), i_batch_tot)

            # writer.add_image('test/L-R-N-M', torch.cat([L_blur[:opt.disp_img_cnt], R_blur[:opt.disp_img_cnt], N_blur[:opt.disp_img_cnt], M_blur[:opt.disp_img_cnt]], 2), i_batch_tot)

            # pdb.set_trace()
            # here snk
            face_region_mask = make_face_region_mask(gt, opt.face_mse_loss_weight, face_region)
            # pdb.set_trace()

        # writer.add_image('debug/guide-blur', torch.cat([guide[:opt.disp_img_cnt], blur[:opt.disp_img_cnt]], 2), i_batch_tot)
        # writer.add_image('debug/face_region_blur', localD_input_real_single[:opt.disp_img_cnt], i_batch_tot)
        # pdb.set_trace()

        # if opt.use_gan_loss and i_batch == batch_num_per_epoch - 1:
            
        #     real_label = real_label[:real_batch_size]
        #     fake_label = fake_label[:real_batch_size]

        # forward
        if opt.just_look:
            with torch.no_grad():
                if opt.only_train_warpnet:
                    warp_guide, grid = GFRNet_warpnet(gt, guide)
                    point_loss = point_crit(grid, lm_gt, lm_mask)
                    tv_loss = tv_crit(grid - orig_xy_map)
                    sym_loss = sym_crit(grid, sym_r)
                    total_loss = opt.point_loss_weight * point_loss + opt.tv_loss_weight * tv_loss + opt.sym_loss_weight * sym_loss
                else:
                    warp_guide, grid, restored_img = GFRNet_G(blur, guide)
                    # TODO: loss ignored

        else:
            if opt.only_train_warpnet:
                warp_guide, grid = GFRNet_warpnet(gt, guide)
            else:
                warp_guide, grid, restored_img = GFRNet_G(blur, guide)
            
            if opt.only_train_warpnet:
                GFRNet_warpnet.zero_grad()
            else:
                # update G
                torch.set_grad_enabled(i_batch % opt.G_update_interval == 0)
                localD_input_fake_single = crop_face_region_batch(restored_img, face_region)

                partD_inputs_left = crop_part_region_batch(warp_guide.detach(), part_pos)
                partD_inputs_real_right = crop_part_region_batch(gt, part_pos)
                partD_inputs_fake_right = crop_part_region_batch(restored_img, part_pos)  # L, R, N, M

                partD_inputs_real_couple = [torch.cat([left, right], 1) for left, right in zip(partD_inputs_left, partD_inputs_real_right)]
                partD_inputs_fake_couple = [torch.cat([left, right], 1) for left, right in zip(partD_inputs_left, partD_inputs_fake_right)]
                local_guide = crop_face_region_batch(warp_guide, face_region)
        
                if opt.use_gan_loss:
                    globalD_input_real_triplet = torch.cat([warp_guide.detach(), guide, gt], 1) # concat on channel
                    globalD_input_fake_triplet = torch.cat([warp_guide.detach(), guide, restored_img], 1)

                # globalD_loss_real = GFRNet_globalD(globalD_input_real_triplet, real_label)
                
                # globalD_loss = (globalD_loss_real + globalD_loss_fake) / 2

                # pdb.set_trace()
                # continue

                # print ('backward & update params!')
                # GFRNet_warpnet.zero_grad()
                GFRNet_G.zero_grad()

                if opt.use_gan_loss:
                    # calc gan loss
                    real_label_val = random.randint(7, 13) / 10
                    real_label = torch.full_like(real_label, real_label_val)
                    # real_label.fill_(real_label_val)
                    output = GFRNet_globalD(globalD_input_fake_triplet)
                    globalD_loss_for_G = globalD_crit(output, real_label)
                    
                    if opt.use_part_gan:
                        partD_losses_for_G = []
                        for part in range(4):
                            real_label_val = random.randint(7, 13) / 10
                            real_label = torch.full_like(real_label, real_label_val)
                            # real_label.fill_(real_label_val)
                            output = GFRNet_partDs[part](partD_inputs_fake_couple[part])
                            partD_losses_for_G.append(partD_crits[part](output, real_label))
                    else:
                        real_label_val = random.randint(7, 13) / 10
                        real_label = torch.full_like(real_label, real_label_val)
                        # real_label.fill_(real_label_val)
                        output = GFRNet_localD(localD_input_fake_single)
                        localD_loss_for_G = localD_crit(output, real_label)

                # pdb.set_trace()
                rec_mse_loss = rec_mse_crit(restored_img, gt, face_region_mask)
                rec_perp_loss = rec_perp_crit(restored_img, gt)
                # rec_perp_loss = None
            
            point_loss = point_crit(grid, lm_gt, lm_mask)
            tv_loss = tv_crit(grid - orig_xy_map)
            sym_loss = sym_crit(grid, sym_r)
            
            with torch.no_grad():
                face_mse_loss_show = rec_face_mse_crit_show(localD_input_fake_single, localD_input_real_single)

            if opt.only_train_warpnet:
                total_loss = opt.point_loss_weight * point_loss + opt.tv_loss_weight * tv_loss + opt.sym_loss_weight * sym_loss
                total_loss.backward()
                optimizer.step()
                
            else: # full net
                if opt.use_gan_loss:
                    rec_loss = opt.rec_mse_loss_weight * rec_mse_loss + opt.rec_perp_loss_weight * rec_perp_loss
                    flow_loss = opt.point_loss_weight * point_loss + opt.tv_loss_weight * tv_loss + opt.sym_loss_weight * sym_loss
                    if opt.use_part_gan:
                        part_gan_loss = opt.partD_loss_weight * (partD_losses_for_G[0] + partD_losses_for_G[1] + partD_losses_for_G[2] + partD_losses_for_G[3])
                        adv_loss =  opt.globalD_loss_weight * globalD_loss_for_G + part_gan_loss
                    else:
                        adv_loss =  opt.globalD_loss_weight * globalD_loss_for_G + opt.localD_loss_weight * localD_loss_for_G
                    
                    total_loss_G = rec_loss + flow_loss + adv_loss
                    
                    # grid.register_hook(print_inter_grad('grid tensor grad to loss:'))

                    #########################################
                    # rec_loss.backward(retain_graph=True)

                    # if not opt.zero_grad_warpnet:
                    #     for param in GFRNet_G.warpNet.parameters():
                    #         # print (param)
                    #         # pdb.set_trace()
                    #         param.grad *= 0.001
                    #     optimizerWarpNet.step()
                    #     optimizerWarpNet.zero_grad()

                    # adv_loss.backward()

                    # if not opt.zero_grad_warpnet:
                    #     for param in GFRNet_G.warpNet.parameters():
                    #         param.grad *= 1
                    #     optimizerWarpNet.step()
                    #     optimizerWarpNet.zero_grad()

                    
                    # flow_loss.backward()
                    # for param in GFRNet_G.warpNet.parameters():
                    #     param.grad *= 1
                    # optimizerWarpNet.step()
                    # optimizerWarpNet.zero_grad()
                    ############################################
                    # total_loss_G = rec_mse_loss
                else:
                    # total_loss_G = opt.point_loss_weight * point_loss + opt.tv_loss_weight * tv_loss + opt.sym_loss_weight * sym_loss\
                    #         + opt.rec_mse_loss_weight * rec_mse_loss + opt.rec_perp_loss_weight * rec_perp_loss
                    total_loss_G = rec_loss + flow_loss

                # grid.register_hook(print_inter_grad('grid tensor grad to globalD loss:'))
                # restored_img.register_hook(print_inter_grad('restored_img tensor grad to rec mse loss:'))

                # print ('global D loss:', globalD_loss_for_G.item())
                # print ('local D loss:', localD_loss_for_G.item())
                # print ('rec mse loss:', rec_mse_loss.item())
                # print ('tot loss G:', total_loss_G)

                if i_batch % opt.G_update_interval == 0:
                    if opt.debug_info:
                        debug_print ('update G ... ')
                        debug_print ('i_batch = %d' % i_batch)
                    total_loss_G.backward()
                    optimizerG.step()
                # optimizerRecNet.step()
                # sym_loss.backward()
                # tv_loss.backward()
                # optimizerG.step()

                if opt.use_gan_loss:
                    # update D
                    torch.set_grad_enabled(i_batch % opt.D_update_interval == 0)
                    real_label_val = random.randint(7, 13) / 10
                    real_label = torch.full_like(real_label, real_label_val)
                    # real_label.fill_(real_label_val)
                    fake_label_val = random.randint(0, 4) / 10
                    fake_label = torch.full_like(fake_label, fake_label_val)
                    # fake_label.fill_(fake_label_val)
                    # pdb.set_trace()
                    # warp_guide, grid, restored_img = GFRNet_G(blur, guide)
                    # localD_input_fake_single = crop_face_region_batch(restored_img, face_region)
                    # globalD_input_real_triplet = torch.cat([warp_guide, guide, gt], 1) # concat on channel
                    # globalD_input_fake_triplet = torch.cat([warp_guide, guide, restored_img], 1)

                    ## update global D
                    GFRNet_globalD.zero_grad()
                    real_output = GFRNet_globalD(globalD_input_real_triplet)
                    globalD_loss_for_D_real = globalD_crit(real_output, real_label)

                    fake_output = GFRNet_globalD(globalD_input_fake_triplet.detach())
                    globalD_loss_for_D_fake = globalD_crit(fake_output, fake_label)
                    
                    globalD_loss_for_D = (globalD_loss_for_D_real + globalD_loss_for_D_fake) / 2


                    if i_batch % opt.D_update_interval == 0:
                        if opt.debug_info:
                            debug_print ('update global D ... ')
                            debug_print ('i_batch = %d' % i_batch)

                        globalD_loss_for_D.backward()
                        optimizerGlobalD.step()

                    if opt.use_part_gan:
                        partD_losses_for_D = []
                        ## update part Ds
                        for part in range(4):
                            real_label_val = random.randint(7, 13) / 10
                            real_label = torch.full_like(real_label, real_label_val)
                            # real_label.fill_(real_label_val)
                            fake_label_val = random.randint(0, 4) / 10
                            fake_label = torch.full_like(fake_label, fake_label_val)
                            # fake_label.fill_(fake_label_val)

                            GFRNet_partD = GFRNet_partDs[part]
                            partD_crit = partD_crits[part]
                            partD_optim = partD_optims[part]

                            GFRNet_partD.zero_grad()
                            real_output = GFRNet_partD(partD_inputs_real_couple[part])
                            partD_loss_for_D_real = partD_crit(real_output, real_label)

                            fake_output = GFRNet_partD(partD_inputs_fake_couple[part].detach())
                            partD_loss_for_D_fake = partD_crit(fake_output, fake_label)

                            partD_loss_for_D = (partD_loss_for_D_real + partD_loss_for_D_fake) / 2

                            partD_losses_for_D.append(partD_loss_for_D.item())
                            
                            if i_batch % opt.D_update_interval == 0:
                                if opt.debug_info and not part:
                                    debug_print ('update part D ... ')
                                    debug_print ('i_batch = %d' % i_batch)

                                partD_loss_for_D.backward()
                                partD_optim.step()
                    else:
                        ## update local D
                        real_label_val = random.randint(7, 13) / 10
                        real_label = torch.full_like(real_label, real_label_val)
                        # real_label.fill_(real_label_val)
                        fake_label_val = random.randint(0, 4) / 10
                        fake_label = torch.full_like(fake_label, fake_label_val)
                        # fake_label.fill_(fake_label_val)

                        GFRNet_localD.zero_grad()
                        real_output = GFRNet_localD(localD_input_real_single)
                        localD_loss_for_D_real = localD_crit(real_output, real_label)

                        fake_output = GFRNet_localD(localD_input_fake_single.detach())
                        localD_loss_for_D_fake = localD_crit(fake_output, fake_label)

                        localD_loss_for_D = (localD_loss_for_D_real + localD_loss_for_D_fake) / 2

                        if i_batch % opt.D_update_interval == 0:
                            if opt.debug_info:
                                debug_print ('update localD ... ')
                                debug_print ('i_batch = %d' % i_batch)

                            localD_loss_for_D.backward()
                            optimizerLocalD.step()

        # pdb.set_trace()

        if opt.save_imgs:            
            if opt.only_train_warpnet:
                save_result_imgs(sample_batched['img_path'], [guide, gt, warp_guide])
            else:
                save_result_imgs(sample_batched['img_path'], [blur, gt, restored_img])
            continue
            
        
        # tensorboardX display
        if i_batch % opt.disp_freq == 0:
            # writer.add_image('guide', guide[:opt.disp_img_cnt], i_batch_tot)
            # writer.add_image('groundtruth', gt[:opt.disp_img_cnt], i_batch_tot)
            # writer.add_image('warp guide', warp_guide[:opt.disp_img_cnt], i_batch_tot)
            if opt.only_train_warpnet:
                writer.add_image('guide-gt-warp', norm_to_01(torch.cat([guide[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], warp_guide[:opt.disp_img_cnt]], 2)), i_batch_tot)
                
                writer.add_scalar('loss/point_loss', point_loss.item(), i_batch_tot)
                writer.add_scalar('loss/tv_loss', tv_loss.item(), i_batch_tot)
                writer.add_scalar('loss/sym_loss', sym_loss.item(), i_batch_tot)
                writer.add_scalar('loss/tot_loss', total_loss.item(), i_batch_tot)
            else:
                writer.add_image('global/guide-warp-gt-blur-restored', norm_to_01(torch.cat([guide[:opt.disp_img_cnt], warp_guide[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], blur[:opt.disp_img_cnt], restored_img[:opt.disp_img_cnt]], 2)), i_batch_tot)
                writer.add_image('local/warp-gt-blur-restored', norm_to_01(torch.cat([local_guide[:opt.disp_img_cnt], localD_input_real_single[:opt.disp_img_cnt], face_region_blur[:opt.disp_img_cnt], localD_input_fake_single[:opt.disp_img_cnt]], 2)), i_batch_tot)
                for idx, part in enumerate(['L', 'R', 'N', 'M']):
                    writer.add_image('part/%s/warp-gt-blur-restored' % part, norm_to_01(torch.cat([ partD_inputs_left[idx][:opt.disp_img_cnt], partD_inputs_real_right[idx][:opt.disp_img_cnt], part_region_blur[idx][:opt.disp_img_cnt], partD_inputs_fake_right[idx][:opt.disp_img_cnt] ], 2)), i_batch_tot)

                writer.add_scalar('loss/G/point_loss', point_loss.item(), i_batch_tot)
                writer.add_scalar('loss/G/tv_loss', tv_loss.item(), i_batch_tot)
                writer.add_scalar('loss/G/sym_loss', sym_loss.item(), i_batch_tot)
                if opt.use_gan_loss:
                    writer.add_scalar('loss/G/globalD_loss_for_G', globalD_loss_for_G.item(), i_batch_tot)
                    if opt.use_part_gan:
                        writer.add_scalar('loss/G/part/L/D_loss_for_G', partD_losses_for_G[0].item(), i_batch_tot)
                        writer.add_scalar('loss/G/part/R/D_loss_for_G', partD_losses_for_G[1].item(), i_batch_tot)
                        writer.add_scalar('loss/G/part/N/D_loss_for_G', partD_losses_for_G[2].item(), i_batch_tot)
                        writer.add_scalar('loss/G/part/M/D_loss_for_G', partD_losses_for_G[3].item(), i_batch_tot)
                    else:
                        writer.add_scalar('loss/G/localD_loss_for_G', localD_loss_for_G.item(), i_batch_tot)
                writer.add_scalar('loss/G/rec_mse_loss', rec_mse_loss.item(), i_batch_tot)
                writer.add_scalar('loss/G/rec_face_mse_loss', face_mse_loss_show.item(), i_batch_tot)

                writer.add_scalar('loss/G/rec_perp_loss', rec_perp_loss.item(), i_batch_tot)
                writer.add_scalar('loss/G/tot_loss_G', total_loss_G.item(), i_batch_tot)
                if opt.use_gan_loss:
                    writer.add_scalar('loss/D/Global/globalD_loss_for_D', globalD_loss_for_D.item(), i_batch_tot)
                    if opt.use_part_gan:
                        writer.add_scalar('loss/D/part/L/D_loss_for_D', partD_losses_for_D[0], i_batch_tot)
                        writer.add_scalar('loss/D/part/R/D_loss_for_D', partD_losses_for_D[1], i_batch_tot)
                        writer.add_scalar('loss/D/part/N/D_loss_for_D', partD_losses_for_D[2], i_batch_tot)
                        writer.add_scalar('loss/D/part/M/D_loss_for_D', partD_losses_for_D[3], i_batch_tot)
                    else:
                        writer.add_scalar('loss/D/Local/localD_loss_for_D', localD_loss_for_D.item(), i_batch_tot)
            

        # pdb.set_trace()
        # print loss
        if i_batch % opt.print_freq == 0:
            if opt.only_train_warpnet:
                print ('Time: %s [(%d/%d) ; (%d/%d)] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f' % 
                    (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i_batch, len(face_dataset_dataloader), \
                    epoch, opt.num_epochs, total_loss.item(), opt.point_loss_weight * point_loss.item(), \
                    opt.tv_loss_weight * tv_loss.item(), opt.sym_loss_weight * sym_loss.item()))
            else:
                gan_loss_info = ""
                if opt.use_gan_loss:
                    if opt.use_part_gan:
                        # part gan losses info see tensorboardX
                        gan_loss_info = "\tGlobal GAN Loss [G/D]=[%f/%f]\t" % (
                            opt.globalD_loss_weight * globalD_loss_for_G.item(),\
                            globalD_loss_for_D.item(),
                        )
                    else:
                        gan_loss_info = "\tGlobal GAN Loss [G/D]=[%f/%f]\tLocal GAN Loss [G/D]=[%f/%f]\t" % (
                            opt.globalD_loss_weight * globalD_loss_for_G.item(),\
                            globalD_loss_for_D.item(),\
                            opt.localD_loss_weight * localD_loss_for_G.item(),\
                            localD_loss_for_D.item()
                        )
                print ('Time: %s [(%s/%d) ; (%d/%d)] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f%s\tRec Mse Loss=%s\tRec Perp Loss=%f\tRec Face Mse Loss=%f' % 
                        (
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),\
                            colored(i_batch, 'green'), batch_num_per_epoch,\
                            epoch, opt.num_epochs,\
                            total_loss_G.item(),\
                            opt.point_loss_weight * point_loss.item(),\
                            opt.tv_loss_weight * tv_loss.item(),\
                            opt.sym_loss_weight * sym_loss.item(),\
                            gan_loss_info,\
                            colored(opt.rec_mse_loss_weight * rec_mse_loss.item(), 'yellow'),
                            opt.rec_perp_loss_weight * rec_perp_loss.item(),
                            opt.rec_mse_loss_weight * face_mse_loss_show.item()
                        )
                )
                
                print ('-' * opt.sep_width)
                    
        
        avg_tv_loss += (opt.tv_loss_weight * tv_loss.item()) / batch_num_per_epoch
        avg_point_loss += (opt.point_loss_weight * point_loss.item()) / batch_num_per_epoch
        avg_sym_loss += (opt.sym_loss_weight * sym_loss.item()) / batch_num_per_epoch
        if not opt.only_train_warpnet:
            if opt.use_gan_loss:
                avg_globalD_loss_for_G += (opt.globalD_loss_weight * globalD_loss_for_G.item()) / batch_num_per_epoch
                avg_globalD_loss_for_D += (globalD_loss_for_D.item()) / batch_num_per_epoch
                if not opt.use_part_gan:
                    avg_localD_loss_for_G += (opt.localD_loss_weight * localD_loss_for_G.item()) / batch_num_per_epoch
                    avg_localD_loss_for_D += (localD_loss_for_D.item()) / batch_num_per_epoch
            avg_rec_mse_loss += (opt.rec_mse_loss_weight * rec_mse_loss.item()) / batch_num_per_epoch
            avg_rec_perp_loss += (opt.rec_perp_loss_weight * rec_perp_loss.item()) / batch_num_per_epoch
            avg_rec_face_mse_loss += (opt.rec_mse_loss_weight * face_mse_loss_show.item()) / batch_num_per_epoch

        i_batch_tot += 1
    
    if not opt.only_train_warpnet:
        writer.add_scalar('loss/G/epoch_avg/rec_mse_loss', avg_rec_mse_loss, epoch)

    if opt.save_imgs:
        print ('save imgs over!')
        break

    # epoch level loss info
    print ()
    print ('=' * opt.sep_width)
    if opt.only_train_warpnet:
        print ('Time: %s Epoch [%d/%d] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f' % 
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, opt.num_epochs, avg_point_loss + avg_tv_loss + avg_sym_loss, avg_point_loss, avg_tv_loss, avg_sym_loss))
    else:
        print ('Time: %s Epoch [%s/%d] Tot Loss = %f\tPoint Loss = %f\tTV Loss=%f\tSym Loss=%f\tGlobal GAN Loss [G/D]=[%f/%f]\tLocal GAN Loss [G/D]=[%f/%f]\tRec Mse Loss=%s\tRec Perp Loss=%f\tRec Face Mse Loss=%f' % 
                        (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), colored(epoch, 'green'), opt.num_epochs, avg_point_loss + avg_tv_loss + avg_sym_loss + avg_globalD_loss_for_G + avg_localD_loss_for_G + avg_rec_mse_loss,\
                        avg_point_loss, avg_tv_loss, avg_sym_loss, avg_globalD_loss_for_G, avg_globalD_loss_for_D, avg_localD_loss_for_G, avg_localD_loss_for_D, colored(avg_rec_mse_loss, 'yellow'), avg_rec_perp_loss, avg_rec_face_mse_loss))
    print ('=' * opt.sep_width)
    print ()
    
    # save model

    if opt.only_train_warpnet:
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
    else:
        if (not opt.just_look) and ((epoch + 1) % opt.save_epoch_freq == 0):
            
            checkpoint_file = path.join(opt.checkpoint_dir, 'checkpoint_%02d.pt' % (epoch+1))
            # print ('epoch is', epoch, 'opt.save_epoch_freq is', opt.save_epoch_freq)
            print ('save model to %s ...' % checkpoint_file)
            save_dict = {
                'epoch': epoch,
                'i_batch_tot': i_batch_tot,
                'use_gan_loss': opt.use_gan_loss,
                # models
                'G_state_dict': GFRNet_G.state_dict(),
                'globalD_state_dict': GFRNet_globalD.state_dict(),
                # optimizers
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerGlobalD_state_dict': optimizerGlobalD.state_dict(),
                # losses
                'tv_loss': avg_tv_loss,
                'point_loss': avg_point_loss,
                'sym_loss': avg_sym_loss,
                
                'globalD_loss_for_G': avg_globalD_loss_for_G,
                'rec_mse_loss': avg_rec_mse_loss,
                'rec_perp_loss': avg_rec_perp_loss,
                'globalD_loss_for_D': avg_globalD_loss_for_D
            }
            if opt.use_gan_loss:
                if opt.use_part_gan:
                    for idx, part in enumerate(['L', 'R', 'N', 'M']):
                        save_dict['partD_%s_state_dict' % part] = GFRNet_partDs[idx].state_dict()
                        save_dict['optimizerPartD_%s_state_dict' % part] = partD_optims[idx].state_dict()
                else:
                    save_dict['localD_state_dict'] = GFRNet_localD.state_dict()
                    save_dict['optimizerLocalD_state_dict'] = optimizerLocalD.state_dict()
                    save_dict['localD_loss_for_D'] = avg_localD_loss_for_D
                    save_dict['localD_loss_for_G'] = avg_localD_loss_for_G

            torch.save(save_dict, checkpoint_file)
        
if not opt.save_imgs:
    writer.close()



def test():
    create_orig_xy_map()  

if __name__ == '__main__':
    # test()
    pass
