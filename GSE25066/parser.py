
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split

import shutil

import os

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path.decode('utf-8')) 
        print path+' 创建成功'
        return True
    else:
        print path+' 目录已存在'
        return False

def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    child_file_name=[]
    full_child_file_list = []
    for allDir in pathDir:
        allDir =unicode(allDir, 'utf-8')   
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        full_child_file_list.append(child)
        child_file_name.append(allDir)
    #return  full_child_file_list, child_file_name
    return child_file_name

# load data
def tolist(f):
    xn = pd.read_csv(f, sep='\t', index_col=0)
    xn = xn.fillna(xn.mean())
    xn = xn.transpose()
    xn = [x[1:] for x in xn.itertuples()]
    return xn

# write each line into a txt file
# write a train/val txt file(s)

# get normal records and return the number of records
def getnormalrecords(xn):
    for eachrow in xn:
        filename = 'normal-' + str(xn.index(eachrow)) + '.txt'
        file_write_normal = open('./Normal/' + filename, 'w')
        eachline_list = [str(x) for x in eachrow]
        eachline = " ".join(eachline_list)
        file_write_normal.writelines(eachline)
        #print 'normal:', eachrow[0]
        file_write_normal.write('\n')
        file_write_normal.close()
        # set normal as 0
        file_label.write(filename + ' ' + str(0))
        file_label.write('\n')
    return len(xn)

# get tumor records and return the number of records
def gettumorrecords(xc):
    for eachrow in xc:
        filename = 'tumor-' + str(xc.index(eachrow)) + '.txt'
        file_write_tumor = open('./Tumor/' + filename, 'w')
        eachline_list = [str(x) for x in eachrow]
        eachline = " ".join(eachline_list)
        file_write_tumor.writelines(eachline)
        #print 'tumor:', eachrow[0]
        file_write_tumor.write('\n')
        file_write_tumor.close()
        # set tumor as 1
        file_label.write(filename + ' ' + str(1))
        file_label.write('\n')
    return len(xc)

if __name__ == '__main__':

    train_ratio = 0.7

    ds = 'GSE25066' #'BC-TCGA' #'GSE2034' #'GSE25066'
    xn = tolist('./' + ds + '-Normal.txt')
    xc = tolist('./' + ds + '-Tumor.txt')

    # generate label.txt and two data dirs
    file_label = open('./label.txt', 'w')
    mkdir(r"./Normal")
    normal_num = getnormalrecords(xn)
    mkdir(r"./Tumor")
    tumor_num = gettumorrecords(xc)
    file_label.close()

    # split those records
    train_num4normal = int(round(normal_num * train_ratio))
    #print train_num4normal
    val_num4normal = normal_num - train_num4normal
    train_num4tumor = int(round(train_num4normal * train_ratio))
    val_num4tumor = val_num4normal

    shutil.rmtree(r"./train")
    shutil.rmtree(r"./val")
    mkdir(r"./train")
    mkdir(r"./val")

    fullnormallist = eachFile(r"./Normal")
    fulltumorlist = eachFile(r"./Tumor")

    # split labels
    train_file_label = open('./train.txt', 'w')
    test_file_label = open('./val.txt', 'w')

    for srcfn in fullnormallist:
        num_txt = srcfn.split('-')[1]
        filenumattached = int(num_txt.split('.')[0])

        #print filenumattached, train_num4normal

        if filenumattached in range(train_num4normal):
            destfn = "./train/"
            train_file_label.write(srcfn)
            train_file_label.write(" 0")
            train_file_label.write("\n")
        else:
            destfn = "./val/"
            test_file_label.write(srcfn)
            test_file_label.write(" 0")
            test_file_label.write("\n")
        shutil.copy("./Normal/" + srcfn, destfn)

    for srcfn in fulltumorlist:
        num_txt = srcfn.split('-')[1]
        filenumattached = int(num_txt.split('.')[0])
        if filenumattached in range(train_num4tumor):
            destfn = "./train/"
            train_file_label.write(srcfn)
            train_file_label.write(" 1")
            train_file_label.write("\n")
        else:
            destfn = "./val/"
            test_file_label.write(srcfn)
            test_file_label.write(" 1")
            test_file_label.write("\n")
        shutil.copy("./Tumor/" + srcfn, destfn)

    train_file_label.close()
    test_file_label.close()

