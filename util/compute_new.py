# -*-coding:utf-8 -*-
'''
@Time    :   2023/09/14 21:51
@Author  :   Jialong Jiang, Yingying Gong
'''

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import os
from sklearn.cluster import KMeans


def onmf_discretize(onmf_rep_ori, fig_folder):

    num_spin = onmf_rep_ori.shape[1]
    sc.set_figure_params(figsize=[2, 2])
    max_num_panels = 21
    fig, grid = sc.pl._tools._panel_grid(0.3, 0.3, ncols=7, num_panels=min(max_num_panels, num_spin))

    onmf_rep_tri = np.zeros(onmf_rep_ori.shape) 
    rec_kmeans = np.zeros(num_spin, dtype=object)
    for ii in range(num_spin):

        km_fit = KMeans(n_clusters=3, n_init=10)
        km_fit.fit(onmf_rep_ori[:, ii].reshape(- 1, 1))

        label_ord = np.argsort(km_fit.cluster_centers_.reshape(- 1))
        onmf_rep_tri[:, ii] = (km_fit.labels_ == label_ord[0]) * (-1) + (km_fit.labels_ == label_ord[2]) * 1
        rec_kmeans[ii] = km_fit

        if ii < max_num_panels:
            ax = plt.subplot(grid[ii])
            plt.plot(np.sort(onmf_rep_ori[:, ii]));
            plt.plot(np.sort(km_fit.cluster_centers_[km_fit.labels_].reshape(- 1)));
        
    if fig_folder:
        plt.savefig(fig_folder + 'onmf_discretize.png', bbox_inches='tight')
        plt.close() # the plot is saved but now shown
    
    
    return onmf_rep_tri

def corr(data):
    return data.T.dot(data) / data.shape[0]

def corr_mean(cur_data):
    rec_data = np.zeros(2, dtype=object)
    rec_data[0] = corr(cur_data)
    rec_data[1] = np.mean(cur_data, axis=0).reshape(- 1, 1)
            
    return rec_data

def sample_corr_mean(samp_full, comp_bin):
    
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)
    
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])
        
    return raw_corr_data, samp_list