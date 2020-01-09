#coding=utf-8
'''
Created on 2018-5-11
'''

import random
import os
import re
import numpy as np
random_seed = ['z','y','x','w','v','u','t','s','r','q','p','o','n','m','l','k','j','i','h','g','f','e','d','c','b','a', 'Z','Y','X','W','V','U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','E','D','C','B','A']
def randomChar(len):
    return ''.join(random.sample(random_seed, len))


def scan_files(directory,prefix=None,postfix=None):
    files_list=[]

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(special_file)
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(special_file)
            else:
                files_list.append(special_file)
                            
    return files_list

def scan_files_no_root(directory,prefix=None,postfix=None,slug=None):
    files_list=[]
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(slug,special_file))
            elif prefix:
                if special_file.startswith(prefix):                                    
                    files_list.append(os.path.join(slug,special_file))
            else:
                files_list.append(os.path.join(slug,special_file))
                            
    return files_list


dir = "../media/trs2/Xray20190723/cut_Image_core_coreless_battery_sub_2000_500/"
targetdir="../media/trs2/Xray20190723/train_test_txt/battery_sub/sub_eval_core_coreless.txt"
imagelist = scan_files_no_root(dir,slug="",postfix=".jpg")
f2 = open(targetdir,'w')

for imageinfo in imagelist: 
    names= os.path.splitext(imageinfo) 
    f2.write(names[0])
    f2.write('\n')

f2.close()