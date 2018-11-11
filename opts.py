import argparse
import os
from os import path
import pdb
from termcolor import colored
import getpass

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--subset', type=str, default='train', help='to use dataset subset')
parser.add_argument('--img_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered', help='img_dir + subset is the real img dir path')
parser.add_argument('--sym_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/sym', help='the dir storing symmetric axis info files')
parser.add_argument('--landmark_dir', type=str, default='/home/snk/FaceCorrespondence/GFRNet_train/datasets/filtered/landmark', help='the dir storing landmarks info files')
parser.add_argument('--num_workers', type=int, default=8, help="the data loader num workers")
parser.add_argument('--img_size', type=int, default=256, help='the image size (current default square)')
parser.add_argument('--part_size', type=int, default=64, help='the part size of eyes, nose and mouth (current default square)')

parser.add_argument('--point_loss_weight', type=float, default=10, help='the point loss weight')
parser.add_argument('--tv_loss_weight', type=float, default=1, help='the tv loss weight')
parser.add_argument('--sym_loss_weight', type=float,default=1, help='the sym loss weight')
parser.add_argument('--globalD_loss_weight', type=float, default=1, help='the global D for G loss weight')
parser.add_argument('--localD_loss_weight', type=float, default=0.5, help='the local D for G loss weight')
# 2 --> 1
parser.add_argument('--partD_loss_weight', type=float, default=1, help='the part D (L, R, N, M) for G loss weight')

parser.add_argument('--rec_mse_loss_weight', type=float, default=100, help='the rec mse loss weight')
parser.add_argument('--rec_perp_loss_weight', type=float, default=0.001, help='the rec perceptual loss weight')

parser.add_argument('--face_mse_loss_weight', type=float, default=1, help='the mse loss weight in face region relative to background(1)')

parser.add_argument('--sym_loss_C_start', type=int, default=10, help='the skip interval C_start of sym loss')
parser.add_argument('--sym_loss_C_step', type=int, default=1, help='the skip interval C_step of sym loss')
parser.add_argument('--sym_loss_C_end', type=int, default=10, help='the skip interval C_end of sym loss')

parser.add_argument('--num_epochs', type=int, default=150, help='the total num of training epochs')



parser.add_argument('--print_freq', type=int, default=10, help='print loss info every X iters')
parser.add_argument('--disp_freq', type=int, default=10, help='refresh the tensorboardX info every X iters')
parser.add_argument('--disp_img_cnt', type=int, default=4, help='the num of displayed images')
parser.add_argument('--save_epoch_freq', type=int, default=50, help='save model every X epochs')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='the dir to save model checkpoints')
parser.add_argument('--save_imgs_dir', type=str, default="save_imgs", help='the dir to save warped image')

# model arch
parser.add_argument('--ngf', type=int, default=64, help='the num of generator(warpNet) 1st conv filters')
parser.add_argument('--output_nc', type=int, default=2, help='the num of generator(warpNet) last conv filters (grid channels)')
parser.add_argument('--output_nc_img', type=int, default=3, help='the num of generator(recNet) output img channels')
parser.add_argument('--input_nc_img', type=int, default=3, help='the num of input image channels for data loading')
parser.add_argument('--load_warpnet_ckpt', type=str, default="./checkpoints/GFRNet/w_sym_loss_C=10/checkpoint_40.pt", help='load the pretrained checkpoint of warpnet')


# parser.add_argument('--exp_name', type=str, default="GFRNet/id=01", help='exp name for tensorboard logs')
# parser.add_argument('--exp_name', type=str, default="GFRNet/w_sym_loss_C=10", help='exp name for tensorboard logs')
parser.add_argument('--exp_name', type=str, default="GFRNet/debug", help='exp name for tensorboard logs')

parser.add_argument('--load_checkpoint', type=str, default=None, help='the dir to load model checkpoints from for continuing training')
parser.add_argument('--just_look', action='store_true', help='just look, dont update model') # still enable tensorboardX for looking
parser.add_argument('--save_imgs', action='store_true', help='use save image or not') # disable tensorboardX, if save_imgs is True, then just_look should also be true

parser.add_argument('--debug_info', action='store_true', help='to show debug info or not')
parser.add_argument('--only_train_warpnet', action='store_true', help='only train warpnet for pretraining')
parser.add_argument('--not_use_gan_loss', action='store_true', help='to use global&local gan loss for recnet')
parser.add_argument('--zero_grad_warpnet', action='store_true', help='whether the grad of recnet backprop to warpnet')
parser.add_argument('--lr_decay_epochs', type=int, default=50, help='decay lr by lr_decay_rate every X epochs')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate every lr_decay_epochs epochs')


parser.add_argument('--use_4x', action='store_true', help='fix downsampler scale to 4x for test')
parser.add_argument('--use_8x', action='store_true', help='fix downsampler scale to 8x for test')
parser.add_argument('--use_part_gan', action='store_true', help='whether to use part gan(conditional) (L,R eyes & nose & mouth) or local gan(unconditional)')
parser.add_argument('--G_update_interval', type=int, default=1, help='update G every X iters')
parser.add_argument('--D_update_interval', type=int, default=1, help='update D every X iters')
parser.add_argument('--sep_width', type=int, default=50, help='print how many separators each line')
parser.add_argument('--flip_prob', type=float, default=0.5, help='the probability to horizontally flip gt & guide for data augmentation')

parser.add_argument('--use_custom_init', action='store_true', help='whether to use custom initialization or pytorch default ones')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--minusone_to_one', action='store_true', help='to normalize the imgs to [-1, 1] or [0, 1] tensors')
parser.add_argument('--use_lsgan', action='store_true', help='whether to use lsgan')
parser.add_argument('--log_imgs_out', action='store_true', help='save training resulting imgs to a separate dir rather than tensorboardX to save disk space')
parser.add_argument('--log_imgs_dir', type=str, default='logs/', help='the root dir to log training resulting imgs out')
parser.add_argument('--log_imgs_epoch_freq', type=int, default=1, help='to log training resulting imgs out every X epochs')
parser.add_argument('--log_imgs_num', type=int, default=5, help='random pick X imgs to log out every log_imgs_epoch_freq epochs')

parser.add_argument('--save_best_model', action='store_true', help='whether to save the best current model evaluted by epoch_avg_rec_mse_loss')
parser.add_argument('--new_start_lr', action='store_true', help='directly use lr rather than scheduler decayed lr')
parser.add_argument('--hpc_version', action='store_true', help='use on HPC servers')



opt = parser.parse_args()

user_name = getpass.getuser()
if user_name == 'zhangwenqiang':
    opt.hpc_version = True
elif user_name == 'snk':
    opt.hpc_version = False
else:
    pass

if opt.hpc_version:
    opt.img_dir = '/home/zhangwenqiang/jobs/GFRNet_pytorch/datasets/filtered'
    opt.sym_dir = '/home/zhangwenqiang/jobs/GFRNet_pytorch/datasets/filtered/sym'
    opt.landmark_dir = '/home/zhangwenqiang/jobs/GFRNet_pytorch/datasets/filtered/landmark'

opt.disp_img_cnt = min(opt.disp_img_cnt, opt.batch_size)
opt.checkpoint_dir = path.join(opt.checkpoint_dir, opt.exp_name)
opt.save_imgs_dir = path.join(opt.save_imgs_dir, opt.exp_name)

if opt.log_imgs_out:
    opt.log_imgs_dir = path.join(opt.log_imgs_dir, opt.exp_name)

opt.use_gan_loss = not opt.not_use_gan_loss
opt.use_custom_init = True

if opt.only_train_warpnet:
    opt.use_gan_loss = False
    opt.load_warpnet_ckpt = None

if not opt.use_gan_loss:
    opt.G_update_interval = 1

if opt.use_part_gan:
    assert opt.use_gan_loss == True, 'Use part gan must enable gan loss!'

if opt.save_imgs:
    opt.just_look = True
    opt.flip_prob = -1

print ('import opts!')

def make_dir(dir_path):
    if not path.exists(dir_path):
        print ('mkdir', dir_path)
        os.makedirs(dir_path)

if not opt.just_look:
    make_dir(opt.checkpoint_dir)
if opt.save_imgs:
    make_dir(opt.save_imgs_dir)

if opt.log_imgs_out:
    make_dir(opt.log_imgs_dir)
    make_dir(path.join(opt.log_imgs_dir, 'epochs'))
    make_dir(path.join(opt.log_imgs_dir, 'epochs/global'))
    make_dir(path.join(opt.log_imgs_dir, 'epochs/local'))
    for part in ['L', 'R', 'N', 'M']:
        make_dir(path.join(opt.log_imgs_dir, 'epochs/parts/%s' % part))


if opt.debug_info:
    if opt.only_train_warpnet:
        print (">>> Only Train WarpNet for Pretraining!")
    if opt.not_use_gan_loss:
        print ('>>> Not Use Gan Loss!')



def show_switches_states():
    switches = ['cuda', 'just_look', 'save_imgs', 'debug_info', 'only_train_warpnet', 'not_use_gan_loss', 'use_gan_loss',\
     'zero_grad_warpnet', 'use_4x', 'use_8x', 'use_part_gan', 'use_custom_init', 'minusone_to_one', 'use_lsgan', 'log_imgs_out']
    dict_opt = vars(opt)
    for switch in switches:
        if dict_opt[switch]:
            print (colored('%s\t✔️' % switch, 'yellow'))
    # pdb.set_trace()

show_switches_states()

    





