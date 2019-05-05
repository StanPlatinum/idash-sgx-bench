import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split

# load data
def tolist(f):
    xn = pd.read_csv(f, sep='\t', index_col=0)
    xn = xn.fillna(xn.mean())
    xn = xn.transpose()
    xn = [x[1:] for x in xn.itertuples()]
    return xn


ds = 'GSE25066' #'BC-TCGA' #'GSE2034' #'GSE25066'
xn = tolist('./' + ds + '/' + ds + '-Normal-50-sgx.txt')
xc = tolist('./' + ds + '/' + ds + '-Tumor-50-sgx.txt')

# write each line into a txt file
# write a train/val txt file(s)

file_label = open('./label.txt', 'w')

for eachrow in xn:
    filename = 'normal-' + str(xn.index(eachrow)) + '.txt'
    file_write_normal = open('./Normal/' + filename, 'w')
    eachline_list = [str(x) for x in eachrow]
    eachline = " ".join(eachline_list)
    # write 3 times
    file_write_normal.writelines(eachline)
    file_write_normal.write('\n')
    file_write_normal.writelines(eachline)
    file_write_normal.write('\n')
    file_write_normal.writelines(eachline)
    
    # print 'normal:', eachrow[0]
    file_write_normal.write('\n')
    file_write_normal.close()
    # set normal as 0
    file_label.write(filename + ' ' + str(0))
    file_label.write('\n')

for eachrow in xc:
    filename = 'tumor-' + str(xc.index(eachrow)) + '.txt'
    file_write_tumor = open('./Tumor/' + filename, 'w')
    eachline_list = [str(x) for x in eachrow]
    eachline = " ".join(eachline_list)
    # write 3 times
    file_write_tumor.writelines(eachline)
    file_write_tumor.write('\n')
    file_write_tumor.writelines(eachline)
    file_write_tumor.write('\n')
    file_write_tumor.writelines(eachline)
    
    # print 'tumor:', eachrow[0]
    file_write_tumor.write('\n')
    file_write_tumor.close()
    # set tumor as 1
    file_label.write(filename + ' ' + str(1))
    file_label.write('\n')

file_label.close()
