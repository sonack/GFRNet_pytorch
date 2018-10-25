import argparse
import os
from os import path
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--subset', type=str, default='train', help='to use dataset subset')
parser.add_argument('--img_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered', help='img_dir + subset is the real img dir path')
parser.add_argument('--sym_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/sym', help='the dir storing symmetric axis info files')
parser.add_argument('--landmark_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/landmark', help='the dir storing landmarks info files')
parser.add_argument('--num_workers', type=int, default=0, help="the data loader num workers")
parser.add_argument('--img_size', type=int, default=256, help='the image size (current default square)')

parser.add_argument('--point_loss_weight', type=float, default=10, help='the point loss weight')
parser.add_argument('--tv_loss_weight', type=float, default=1, help='the tv loss weight')
parser.add_argument('--sym_loss_weight', type=float,default=1, help='the sym loss weight')

parser.add_argument('--sym_loss_C_start', type=int, default=10, help='the skip interval C_start of sym loss')
parser.add_argument('--sym_loss_C_step', type=int, default=1, help='the skip interval C_step of sym loss')
parser.add_argument('--sym_loss_C_end', type=int, default=10, help='the skip interval C_end of sym loss')

parser.add_argument('--num_epochs', type=int, default=100, help='the total num of training epochs')
# parser.add_argument('--exp_name', type=str, default="GFRNet/id=01", help='exp name for tensorboard logs')
parser.add_argument('--exp_name', type=str, default="GFRNet/just_look", help='exp name for tensorboard logs')

parser.add_argument('--print_freq', type=int, default=10, help='print loss info every X iters')
parser.add_argument('--disp_freq', type=int, default=10, help='refresh the tensorboardX info every X iters')
parser.add_argument('--disp_img_cnt', type=int, default=4, help='the num of displayed images')
parser.add_argument('--save_epoch_freq', type=int, default=10, help='save model every X epochs')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='the dir to save model checkpoints')
parser.add_argument('--load_checkpoint', type=str, default=None, help='the dir to load model checkpoints from for continuing training')
parser.add_argument('--just_look', action='store_true', help='just look, dont update model')
# model arch
parser.add_argument('--ngf', type=int, default=64, help='the num of generator(warpNet) 1st conv filters')
parser.add_argument('--output_nc', type=int, default=2, help='the num of generator(warpNet) last conv filters (grid channels)')


opt = parser.parse_args()

opt.disp_img_cnt = min(opt.disp_img_cnt, opt.batch_size)
opt.checkpoint_dir = path.join(opt.checkpoint_dir, opt.exp_name)

print ('import opts!')
if not path.exists(opt.checkpoint_dir):
    print ('mkdir', opt.checkpoint_dir)
    os.makedirs(opt.checkpoint_dir)


