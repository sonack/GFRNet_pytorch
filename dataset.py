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
from custom_transforms import ToTensor

from math import floor, ceil

from opts import opt
# DEBUG switches
DEBUG_INFO = False 
DEBUG_TEST = True


img_suffixes = ['.png']
img_size = opt.img_size

def file_suffix(filename):
    return path.splitext(filename)[-1]

# [1,256] --> [-1, 1]
def normalize(num):
    return (num - 1) / 255 * 2 - 1

# [-1, 1] --> [1,256]
def denormalize(num):
    return (num + 1) / 2 * 255 + 1

Point = namedtuple('Point', ['x', 'y'])

class FaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, subset, landmark_dir, sym_dir, img_dir, transform=None):
        """
        Args:
            subset (string): train, val, test
            landmark_dir (string): Directory with all landmark annotations.
            sym_dir (string):  Directory with all symmetric axis annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmark_dir = landmark_dir
        self.sym_dir = sym_dir
        self.img_dir = path.join(img_dir, subset)
        self.img_list = []
        all_files = os.listdir(path.join(img_dir, subset))

        for filename in all_files:
            full_filename = path.join(self.img_dir, filename)
            # pdb.set_trace()
            if path.isfile(full_filename) and (file_suffix(filename) in img_suffixes):
                self.img_list.append(filename)
        
        if DEBUG_INFO:
            print ('Train img filenames:\n')
            for filename in self.img_list[:10]:
                print (filename)
         
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    # 左上角 (x1,y1) 右下角 (x2, y2)
    # 坐标原点 左上角
    def parse_face_region(self, filename):
        file_id, _, x1, x2, y1, y2 = filename.split('_')
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2.split('.')[0])
        return file_id + '.png.txt', [Point(x1,y1), Point(x2,y2)]

       
    
    def parse_sym_axis(self, filename):
        with open(filename, 'r') as f:
            x_l, y_l, x_r, y_r = list(map(float, f.read().split()))
        return (x_l, y_l), (x_r, y_r)
    
    # 左 gt, 右 guided 右-->左
    def parse_landmark_file(self, filename):
        lm_l = []
        lm_r = []

        top_y = 257
        bottom_y = -1
        left_x = 257
        right_x = -1
        with open(filename, 'r') as f:
            for line in f.readlines():
                # print (line)
                # x1, y1 belong to [1.0,255.0] 
                # x2, y2 belong to [-1.0, 1.0]
                x1, y1, x2, y2 = list(map(float, line.split()))
                top_y = min(top_y, y1)
                bottom_y = max(bottom_y, y1)
                left_x = min(left_x, x1)
                right_x = max(right_x, x1)
                # x1 = normalize(x1)
                # y1 = normalize(y1)
                lm_l.append((x1, y1))
                lm_r.append((x2, y2))
        

    
        assert len(lm_l) == 68 and len(lm_l) == len(lm_r), 'Landmarks length must be 68!'
        return lm_l, lm_r, [Point(round(left_x), round(top_y)), Point(round(right_x), round(bottom_y))]

    # ratio: (up, down, left, right)
    def approx_face_region(self, face_region, ratio=(0.5, 0.05, 0.1, 0.1)):
        p1, p2 = face_region
        w = p2.x - p1.x
        h = p2.y - p1.y
        up_offset = round(h * ratio[0])
        down_offset = round(h * ratio[1])
        left_offset = round(w * ratio[2])
        right_offset = round(w * ratio[3])

        return [
                Point(max(p1.x - left_offset, 0), max(p1.y - up_offset, 0)),
                Point(min(p2.x + right_offset, img_size), min(p2.y + down_offset, img_size))
        ]


    def create_lm_gt_mask(self, lm_l, lm_r):
        lm_gt = np.zeros((2, img_size, img_size), dtype=np.float32)
        lm_mask = np.zeros((1, img_size, img_size), dtype=np.float32)
        # print ('lm_gt', lm_gt.dtype)
        # print ('lm_mask', lm_mask.dtype)

        for id in range(68):
            x1, y1 = lm_l[id]
            x2, y2 = lm_r[id]
            # x1, y1 = denormalize(x1), denormalize(y1)

            # print (x1, y1, x2, y2)

            floor_x1 = floor(x1) - 1
            ceil_x1 = ceil(x1) - 1
            floor_y1 = floor(y1) - 1
            ceil_y1 = ceil(y1) - 1

            if ceil_x1 > 255 or ceil_y1 > 255 or floor_x1 < 0 or floor_y1 < 0:
                print ('skip landmark %d ... ' % id)
                continue
            
            # the 1st channel is x plane
            # the 2nd channel is y plane
            lm_gt[0][floor_y1][floor_x1] = x2
            lm_gt[0][floor_y1][ceil_x1] = x2
            lm_gt[0][ceil_y1][floor_x1] = x2
            lm_gt[0][ceil_y1][ceil_x1] = x2

            lm_gt[1][floor_y1][floor_x1] = y2
            lm_gt[1][floor_y1][ceil_x1] = y2
            lm_gt[1][ceil_y1][floor_x1] = y2
            lm_gt[1][ceil_y1][ceil_x1] = y2

            lm_mask[0][floor_y1][floor_x1] = 1
            lm_mask[0][floor_y1][ceil_x1] = 1
            lm_mask[0][ceil_y1][floor_x1] = 1
            lm_mask[0][ceil_y1][ceil_x1] = 1
            
        return lm_gt, lm_mask





    def __getitem__(self, idx):
        full_filename = path.join(self.img_dir, self.img_list[idx])

        file_id, face_region = self.parse_face_region(self.img_list[idx])

        lm_path = path.join(self.landmark_dir, file_id)
        sym_path = path.join(self.sym_dir, file_id)


        # pdb.set_trace()

        lm_l, lm_r, face_region_calc = self.parse_landmark_file(lm_path)
        sym_l, sym_r = self.parse_sym_axis(sym_path)


        image = io.imread(full_filename)
        # print ('image type', image.dtype)
        wd = int(image.shape[1] // 2)
        left = image[ : , : wd , : ]
        right = image[ : , wd : , : ]

        landmark_left = np.array(lm_l, dtype=np.float32)
        landmark_right = np.array(lm_r, dtype=np.float32)
        sym_l, sym_r = np.array(sym_l, dtype=np.float32), np.array(sym_r, dtype=np.float32)

        lm_gt, lm_mask = self.create_lm_gt_mask(landmark_left, landmark_right)
        
        sample = {
            'img_l': left,
            'lm_l': landmark_left,
            'sym_l': sym_l,

            'img_r': right,
            'lm_r': landmark_right,
            'sym_r': sym_r,

            'face_region': face_region,
            'face_region_calc': face_region_calc,   # left lm calc
            'face_region_approx': self.approx_face_region(face_region_calc),
            'img_path': full_filename,
            
            'lm_gt': lm_gt,
            'lm_mask': lm_mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def test():
    subset = 'train'
    landmark_dir = '/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/landmark'
    sym_dir = '/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/sym'
    img_dir = '/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered'
    face_dataset = FaceDataset(subset, landmark_dir, sym_dir, img_dir)

    print ('Dataset size:', len(face_dataset))
    # print (face_dataset[0])

    # pdb.set_trace()
    
    pos = 'r'
    idx = 1
    sample = face_dataset[idx]
    # print ('img path:', face_dataset[idx]['img_path'])

    lm_l = sample['lm_%c' % pos]
    face_region = sample['face_region']

    p1, p2 = face_region
    print ("face region:", p1, p2)

    p3, p4 = sample['face_region_calc']
    print ("calc face region:", p3, p4)

    p1, p2 = p3, p4

    p5, p6 = sample['face_region_approx']
    print ("approx face region:", p5, p6)

    p1, p2 = p5, p6

    w = p2.x - p1.x
    h = p2.y - p1.y

    fig, ax = plt.subplots(1,1)
    # 左上角坐标 (宽,高)
    # 坐标原点 左上角
    rect = patches.Rectangle((p1.x,p1.y),w,h,linewidth=1,edgecolor='green',facecolor='none')
    ax.add_patch(rect)

    ax.imshow(sample['img_%c' % pos])
    if pos == 'l':
        ax.scatter(lm_l[ : , 0 ], lm_l[ : , 1 ], s=10, c='r', marker='x')
    else:
        ax.scatter(denormalize(lm_l[ : , 0 ]), denormalize(lm_l[ : , 1 ]), s=10, c='r', marker='x')

    sym_l = sample['sym_r']

    # x, y = sym_l
    # if x > y:
    #     y = 256 / x * y
    #     x = 256
    # else:
    #     x = 256 / y * x
    #     y = 256
    # # pdb.set_trace()
    # plt.plot([128, 128+x], [256, 256-y])
    # plt.show()
    print (sample['img_path'])
    print ('sym axis:', sample['sym_r'])
    plt.savefig('result')


if __name__ == '__main__':
    if DEBUG_TEST:
        test()
