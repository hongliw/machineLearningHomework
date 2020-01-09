# -*- coding:utf-8 -*-
"""数据增强
   1. 缩放大小 
   2. 调整亮度与对比度 
   3. 水平翻转 
   4. 垂直翻转 
   5. 水平垂直翻转
   6. 高斯模糊
"""
# 导入库
import cv2
import numpy as np
import os
from shutil import copyfile
anno_dir_path = "media/trs2/Xray20190723/Anno_core_coreless_battery_sub_2000_500/"
img_dir_path = "media/trs2/Xray20190723/cut_Image_core_coreless_battery_sub_2000_500/"

img_names = os.listdir(img_dir_path)
for img_name in img_names:
    # print(img_name)
    if img_name.split('.')[1] != "DS_Store":
        # 读取annotation文件
        bndbox = []
        with open(anno_dir_path + img_name.split('.')[0]+".txt", "r", encoding='utf-8') as f1:
            dataread = f1.readlines()

        for annotation in dataread:
            temp = annotation.split()
            name = temp[0]
            label = temp[1]
             # 只读两类
            if label != '带电芯充电宝' and label != '不带电芯充电宝':
                continue
            xmin = int(temp[2])
            ymin = int(temp[3])
            xmax = int(temp[4])
            ymax = int(temp[5])
            bndbox.append(name)
            bndbox.append(label)
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
        # print(bndbox)
        img = cv2.imread(img_dir_path + img_name) #读入图片
        height,width = img.shape[:2] #获取图片的高和宽
        # print(height, width)

        # 将图像缩小为原来的0.5倍 
        # cv2.resize(变量 ,(宽,高), 插值方法)  
        # img_turn = cv2.resize(img, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_CUBIC)
        #         # cv2.resize(变量 ,(宽,高), 插值方法)   
        # cv2.imwrite(img_dir_path + img_name.split('.')[0] + '_resize.jpg',img_turn)
        # with open(anno_dir_path + img_name.split('.')[0] + '_resize.txt', "w") as f:
        #     line = bndbox[0] + " " + bndbox[1] + " " + str((int)(bndbox[2] / 2)) + " " + str((int)(bndbox[3] / 2)) + " " + str((int)(bndbox[4] / 2)) + " " + str((int)(bndbox[5] / 2))
        #     f.write(line)
        # f.close()

        # 调整亮度与对比度
        # contrast = 1        #对比度
        # brightness = 100    #亮度
        # # cv2.addWeighted(对象,对比度,对象,对比度)
        # img_turn = cv2.addWeighted(img,contrast,img,0,brightness)
        #         #cv2.addWeighted(对象,对比度,对象,对比度)
        # cv2.imwrite(img_dir_path + img_name.split('.')[0] + '_weight.jpg',img_turn)
        # copyfile(anno_dir_path + img_name.split('.')[0]+".txt", anno_dir_path + img_name.split('.')[0]+"_weight.txt")

        # flip(img,1)#1代表水平方向旋转180度
        # flip(img,0)#0代表垂直方向旋转180度
        # flip(img,-1)#-1代表垂直和水平方向同时旋转
        #水平翻转
        img_turn = cv2.flip(img, 1)
        cv2.imwrite(img_dir_path + img_name.split('.')[0]  + '_horizontal.jpg',img_turn)
        with open(anno_dir_path + img_name.split('.')[0] + '_horizontal.txt', "w") as f:
            line = bndbox[0] + " " + bndbox[1] + " " + str(width - bndbox[4]) + " " + str(bndbox[3]) + " " + str(width - bndbox[2]) + " " + str(bndbox[5])
            f.write(line)
        f.close()


        #垂直翻转
        img_turn = cv2.flip(img, 0)
        cv2.imwrite(img_dir_path + img_name.split('.')[0]  + '_vertical.jpg',img_turn)
        with open(anno_dir_path + img_name.split('.')[0] + '_vertical.txt', "w") as f:
            line = bndbox[0] + " " + bndbox[1] + " " + str(bndbox[2]) + " " + str(height - bndbox[5]) + " " + str(bndbox[4]) + " " + str(height - bndbox[3])
            f.write(line)
        f.close()

        
        # 水平垂直翻转
        img_turn = cv2.flip(img, -1)
        cv2.imwrite(img_dir_path + img_name.split('.')[0]  + '_horizontal_vertical.jpg',img_turn)
        with open(anno_dir_path + img_name.split('.')[0] + '_horizontal_vertical.txt', "w") as f:
            line = bndbox[0] + " " + bndbox[1] + " " + str(width - bndbox[4]) + " " + str(height - bndbox[5]) + " " + str(width - bndbox[2]) + " " + str(height - bndbox[3])
            f.write(line)
        f.close()


        # 高斯模糊
        # cv2.GaussianBlur(图像，卷积核，标准差）
        # img_turn = cv2.GaussianBlur(img, (7,7), 1.5)
        # cv2.imwrite(img_dir_path + img_name.split('.')[0]  + '_gaussian.jpg',img_turn)
        # copyfile(anno_dir_path + img_name.split('.')[0]+".txt", anno_dir_path + img_name.split('.')[0]+"_gaussian.txt")