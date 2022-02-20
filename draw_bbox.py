#coding=utf-8
#python 3
import os
import shutil
import cv2
import numpy as np

root_ = "/home/yuyang/data/ICDAR_2015/"
gt_path = root_ + "label/"
img_path = root_ + "image/"

root_save = root_ + "ch4_training_images_draw/"
os.mkdir(root_save)
txt_list = os.listdir(gt_path)

cnt_ = 0
for file_name in txt_list:
    cnt_ += 1
    img_name = file_name.replace('gt_','')
    img_name = img_name.replace('.txt','.jpg')
    print("%d:::%s"%(cnt_,img_name))
    path_img = img_path + img_name
    img = cv2.imread(path_img)
    path_txt = gt_path + file_name
    with open(path_txt,'r',encoding='utf-8') as f:
        str = f.readlines()
    ll_pt = []
    for path in str:
        #print (path.strip())
        path = path.strip()
        list_str = path.encode('utf-8').decode('utf-8-sig').split(',')
        l_pt = []
        l_pt_tmp = []
        for i in range(0,8):
            l_pt.append(int(list_str[i]))
            if i % 2 != 0:
                l_pt_tmp.append(l_pt)
                l_pt = []
        ll_pt.append(l_pt_tmp)

    for v_pt in ll_pt:
        #print(v_pt)
        point = np.array(v_pt,np.int32)
        cv2.polylines(img, [point], True, (0, 255, 255))

    # cv2.imshow('helo',img)
    # cv2.waitKey(0)
    draw_img_path = root_save + img_name
    cv2.imwrite(draw_img_path,img)
