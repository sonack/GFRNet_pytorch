import os
# n0024490002_L_96_74_27_R_172_71_25_N_0_0_0_NewN_139_121_28_M_137_163_28_NewBB_0_0_0_0.jpg
info_name_imgs_dir = '/home/snk/FaceCorrespondence/GFRNet_train/train/GRMouthVGG2/train'
target_dir = ''
# test_str = 'n0024490002_L_96_74_27_R_172_71_25_N_0_0_0_NewN_139_121_28_M_137_163_28_NewBB_0_0_0_0.jpg'
# print (test_str.split('_'))
for idx, filename in enumerate(os.listdir(info_name_imgs_dir)):
    print (filename)
    if idx > 0:
        break
    splited = filename.split('_')
    file_id = splited[0]
    L = [float(splited[2]), float(splited[3]), float(splited[4])]
    R = [float(splited[6]), float(splited[7]), float(splited[8])]
    N = [float(splited[14]), float(splited[15]), float(splited[16])]
    M = [float(splited[18]), float(splited[19]), float(splited[20])]

    print (L,R,N,M)


# 因为gt和guide交换了位置, 所以必须重新计算part位置, 在dataset.py中修改。

