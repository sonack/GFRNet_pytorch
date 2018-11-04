import shutil
import os
import glob
from PIL import Image
import pdb

# SUBSET = "need_analysis"
SUBSET = "better"
better_filelist = "%s.txt" % SUBSET
src_sym_dir = "/home/snk/FaceCorrespondence/GFRNet_train/pytorch/save_imgs/GFRNet/epoch=40_sym"
src_nosym_dir = "/home/snk/FaceCorrespondence/GFRNet_train/pytorch/save_imgs/GFRNet/epoch=40_no_sym"
dest_dir = "%s" % SUBSET
if not os.path.exists(dest_dir):
    print ('mkdir %s.' % dest_dir)
    os.mkdir(dest_dir)

with open(better_filelist, 'r') as f:
    files = f.readlines()

sym_files = [os.path.join(src_sym_dir, file.strip()) for file in files]
nosym_files = [os.path.join(src_nosym_dir, file.strip()) for file in files]

for file, no_sym_file, sym_file in zip(files, nosym_files, sym_files):
    no_sym_img = Image.open(no_sym_file)
    sym_img = Image.open(sym_file)
    new_img = Image.new('RGB', (256*4+2*5, 256+2*2))
    no_sym_img_crop = no_sym_img.crop((0,0,256*3+2*3, 256+2*2))
    sym_img_crop = sym_img.crop((256*2+2*2, 0, 256*3+2*4, 256+2*2))
    new_img.paste(no_sym_img_crop, (0,0))
    new_img.paste(sym_img_crop, (256*3+2*3,0))
    new_img.save(os.path.join(dest_dir, file.strip()))
    print ('save to', os.path.join(dest_dir, file.strip()))
    # pdb.set_trace()
    


