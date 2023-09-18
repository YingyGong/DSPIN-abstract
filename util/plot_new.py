# -*-coding:utf-8 -*-
'''
@Time    :   2023/09/18 11:10
@Author  :   Jialong Jiang, Yingying Gong
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import csv 

def onmf_to_csv(features, gene_name, file_path, thres=0.01):

    num_spin = features.shape[0]
    rec_max_gene = 0
    with open(file_path,'w', newline='') as file:
        write = csv.writer(file)
        for spin in range(num_spin):
            num_gene_show = np.sum(features[spin, :] > thres)
            rec_max_gene = max(rec_max_gene, num_gene_show)
            gene_ind = np.argsort(- features[spin, :])[: num_gene_show]
            cur_line = list(gene_name[gene_ind])
            cur_line.insert(0, spin)
            write.writerow(cur_line) 

    pd_csv = pd.read_csv(file_path, names=range(rec_max_gene + 1))
    pd_csv = pd_csv.transpose()
    pd_csv.to_csv(file_path, header=False, index=False)
    return(file_path)