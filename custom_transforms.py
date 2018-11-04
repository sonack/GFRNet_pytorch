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
import cv2
import numpy as np
from opts import opt
import random

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    def __call__(self, sample):
        sample['img_l'], sample['img_r'], sample['gt'] = self.to_tensor(sample['img_l']), self.to_tensor(sample['img_r']), self.to_tensor(sample['gt'])
        return sample


class NormalizeToMinusOneToOne(object):
    def __init__(self, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
        self.normalize = transforms.Normalize(mean = mean, std = std)
    def __call__(self, sample):
        self.normalize(sample['img_l'])
        self.normalize(sample['img_r'])
        self.normalize(sample['gt'])
        return sample
        
# random horizontal flip 
# 需要处理:
#   - 水平翻转img_l, img_r
#   - 处理 landmarks
#   - 处理 sym

# class Flip(object):
#     def __init__(self, flip_prob = 0.5):
#         self.flip_prob = flip_prob  
#   
#     def __call__(self, sample):
#         pass



DEBUG_INFO = False
# DEBUG_INFO = opt.debug_info

# 各种图像退化变换
# 只退化gt, 左gt 右guide
class GaussianBlur(object):
    def __init__(self, sigma=3, size=13):
        assert isinstance(sigma, (int, float))
        self.sigma = sigma
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.size = (size, size) # size must be odd
        else:
            assert len(size) == 2, "len(size) of GaussianBlur must be 2!"
            self.size = size
    
    def __call__(self, sample):
        if DEBUG_INFO:
            print ('GaussianBlur(sigma=%s, size=%s)' % (self.sigma, self.size))
        if self.sigma > 0:  # 0 indicate no blurring
            gt = sample['img_l']
            sample['img_l'] = cv2.GaussianBlur(gt, self.size, self.sigma)
        return sample


class DownSampler(object):
    def __init__(self, scale):
        # print ("type scale", type(scale))
        assert isinstance(scale, (int, float))
        self.scale = scale
    
    def __call__(self, sample):
        if DEBUG_INFO:
            print ('DownSampler(scale=%s)' % self.scale)
        if self.scale > 1:
            gt = sample['img_l']
            # print ("gt.shape", gt.shape)
            # pdb.set_trace()
            h, w, _ = gt.shape
            scaled_h, scaled_w = int(h / self.scale), int(w / self.scale)
            # print (scaled_w, scaled_h)
            # downsample + upsample
            sample['img_l'] = cv2.resize(gt, (scaled_w, scaled_h), interpolation = cv2.INTER_CUBIC)
        return sample
        
# upsample to (opt.img_size, opt.img_size)
class UpSampler(object): 
    def __init__(self, scale):
        assert isinstance(scale, (int, float))
        self.scale = scale

    def __call__(self, sample):
        if DEBUG_INFO:
            print ('UpSampler()')
        if self.scale > 1:
            gt = sample['img_l']
            sample['img_l'] = cv2.resize(gt, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)
        return sample

class AWGN(object):
    def __init__(self, level):
        assert isinstance(level, (int, float))
        # noise level
        self.level = level

    def __call__(self, sample):
        if DEBUG_INFO:
            print ('AWGN(level=%s)' % self.level)
        if self.level > 0:
            gt = sample['img_l']
            noise = np.random.randn(*gt.shape) * self.level
            # print ("noise shape:", noise.shape)
            # pdb.set_trace()
            # clip(0,255) 防止负数变为255
            sample['img_l'] = (gt + noise).clip(0,255).astype(np.uint8) # otherwise would be np.float64
        return sample
       
# jpeg compressor + decompressor
class JPEGCompressor(object):
    def __init__(self, quality):
        assert isinstance(quality, (int, float))
        self.quality = quality
    
    def __call__(self, sample):
        if DEBUG_INFO:
            print ('JPEGCompressor(quality=%s)' % self.quality)
        if self.quality > 0:    # 0 indicating no lossy compression (i.e losslessly compression)
            gt = sample['img_l']
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            sample['img_l'] = cv2.imdecode(cv2.imencode('.jpg', gt, encode_param)[1], 1)
        return sample
    
class DegradationModel(object):
    def __init__(self):
        self.gaussianBlur_sigma_list = [1 + x * 0.1 for x in range(21)]
        self.gaussianBlur_sigma_list += len(self.gaussianBlur_sigma_list) * [0] # 1/2 trigger this degradation
        self.gaussianBlur_size_list = list(range(3,14,2))
        self.downsample_scale_list = [1 + x * 0.1 for x in range(1,71)]
        self.downsample_scale_list += len(self.downsample_scale_list) * [1]
        if opt.use_4x:
            self.downsample_scale_list = [4]
        if opt.use_8x:
            self.downsample_scale_list = [8]
        self.awgn_level_list = list(range(1, 8, 1))
        self.awgn_level_list += len(self.awgn_level_list) * [0]
        # self.awgn_level_list = [0]
        self.jpeg_quality_list = list(range(10, 40, 1))
        self.jpeg_quality_list += len(self.jpeg_quality_list) * [0]
        # self.jpeg_quality_list = [0]

        
        # ops
        self.gaussianBlur = GaussianBlur(random.choice(self.gaussianBlur_sigma_list), random.choice(self.gaussianBlur_size_list))
        self.downSampler = DownSampler(random.choice(self.downsample_scale_list))
        self.upSampler = UpSampler(self.downSampler.scale)
        self.awgn = AWGN(random.choice(self.awgn_level_list))
        self.jpegCompressor = JPEGCompressor(random.choice(self.jpeg_quality_list))
    
    def random_params(self):
        self.gaussianBlur.sigma = random.choice(self.gaussianBlur_sigma_list)
        self.gaussianBlur.size = (random.choice(self.gaussianBlur_size_list),) * 2

        self.downSampler.scale = random.choice(self.downsample_scale_list)
        self.upSampler.scale = self.downSampler.scale
        self.awgn.level = random.choice(self.awgn_level_list)
        self.jpegCompressor.quality = random.choice(self.jpeg_quality_list)


    def __call__(self, sample):
        self.random_params()
        return self.upSampler(self.jpegCompressor(self.awgn(self.downSampler(self.gaussianBlur(sample)))))

# test_img = '/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/train/n0039900005_NewBB_63_211_62_222.png'
test_img = './first_img.png'

def test_GaussianBlur():
    gaussianBlur = GaussianBlur(3, (7,7))
    img = cv2.imread(test_img)
    cv2.imwrite("blured_leimu.jpg", gaussianBlur({'img_l':img})['img_l'])

def test_DownUpsampler():
    downSampler = DownSampler(2)
    upSampler = UpSampler()
    img = cv2.imread(test_img)
    cv2.imwrite("downupsampled_leimu.jpg", upSampler( {'img_l':downSampler({'img_l':img}) })['img_l'])

def test_AWGN():
    awgn = AWGN(10)
    img = cv2.imread(test_img)
    cv2.imwrite("awgned_leimu.jpg", awgn({'img_l':img})['img_l'])

def test_JPEGCompressor():
    jpegCompressor = JPEGCompressor(1)
    img = cv2.imread(test_img)
    cv2.imwrite("jpeged_leimu.jpg", jpegCompressor({'img_l':img})['img_l'])

def test_DegradationModel():
    degradationModel = DegradationModel()
    # img = cv2.imread(test_img)
    img = io.imread(test_img)
    # pdb.set_trace()
    img_l = img[:,:256,:]
    img_r = img[:,256:,:]
    cv2.imwrite("degraded_leimu.jpg", degradationModel({'img_l':img_l, 'img_r':img_r})['img_l'])
    degraded_sample = degradationModel({'img_l':img_l, 'img_r':img_r})
    degraded_img = degraded_sample['img_l']
    toTensor = ToTensor()
    tensor_sample = toTensor(degraded_sample)
    tensor_img = tensor_sample['img_l']
    # pdb.set_trace()

def test():
    # test_GaussianBlur()
    # test_DownUpsampler()
    # test_AWGN()
    # test_JPEGCompressor()
    test_DegradationModel()


if __name__ == '__main__':
    test()
        
        
    




