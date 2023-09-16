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
import scipy.io as sio


def category_balance_number(total_sample_size, cluster_count, method, maximum_sample_rate):
    if method not in ['equal', 'proportional', 'squareroot']:
        raise ValueError('method must be one of equal, proportional, squareroot')
    
    if method == 'squareroot':
        esti_size = (np.sqrt(cluster_count) / np.sum(np.sqrt(cluster_count)) * total_sample_size).astype(int)
        weight_fun = np.min([esti_size, maximum_sample_rate * np.array(cluster_count)], axis=0)
    elif method == 'equal':
        esti_size = total_sample_size / len(cluster_count)
        weight_fun = np.min([esti_size * np.ones(len(cluster_count)), maximum_sample_rate * np.array(cluster_count)], axis=0)
    else:
        weight_fun = cluster_count

    sampling_number = (weight_fun / np.sum(weight_fun) * total_sample_size).astype(int)
    return sampling_number



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

def para_moments(j_mat, h_vec):
    """
    Calculate the mean and correlation given j network and h vectors"""
    num_spin = j_mat.shape[0]
    num_sample = 3 ** num_spin

    sample_indices = np.indices((3,) * num_spin)
    ordered_sample = (sample_indices - 1).reshape(num_spin, num_sample)

    # Calculate ordered energy and partition function
    j_mat = j_mat + np.diag(np.diag(j_mat))
    ordered_energy = - (h_vec.T @ ordered_sample + np.sum((j_mat @ ordered_sample) * ordered_sample, axis=0) / 2)

    ordered_exp = np.exp(-ordered_energy)
    partition = np.sum(ordered_exp)
    freq = ordered_exp / partition

    mean_para = np.sum(ordered_sample * freq.reshape(1, - 1), axis=1)

    corr_para = np.einsum('i,ji,ki->jk', freq.flatten(), ordered_sample, ordered_sample)

    return corr_para, mean_para

from collections import deque

def compute_gradient(cur_j, cur_h, raw_data, method, train_dat):

    num_spin, num_round = cur_h.shape

    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))
    
    if method == 'pseudo_likelihood':
        pass
    elif method == 'maximum_likelihood':
        for kk in range(num_round):
            corr_para, mean_para = para_moments(cur_j, cur_h[:, kk])
            rec_jgrad[:, :, kk] = corr_para - raw_data[kk][0]
            rec_hgrad[:, kk] = mean_para - raw_data[kk][1].flatten()
    elif method == 'mcmc_maximum_likelihood':
        pass

    return rec_jgrad, rec_hgrad

def apply_regularization(rec_jgrad, rec_hgrad, cur_j, cur_h, train_dat):

    num_spin, num_round = cur_h.shape

    lambda_l1_j, lambda_l1_h, lambda_l2_j, lambda_l2_h, lambda_prior_h = (train_dat.get(key, 0) for key in ["lambda_l1_j", "lambda_l1_h", "lambda_l2_j", "lambda_l2_h", "lambda_prior_h"])

    if lambda_l1_j > 0:
        rec_jgrad += lambda_l1_j * (cur_j / 0.02).clip(-1, 1).reshape(num_spin, num_spin, 1)
    if lambda_l1_h > 0:
        rec_hgrad += lambda_l1_h * (cur_h / 0.02).clip(-1, 1)

    if lambda_l2_j > 0:
        rec_jgrad += lambda_l2_j * cur_j
    if lambda_l2_h > 0:
        rec_hgrad += lambda_l2_h * cur_h

    return rec_jgrad, rec_hgrad

def update_adam(gradient, m, v, counter, stepsz, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** counter)
    v_hat = v / (1 - beta2 ** counter)
    update = stepsz * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m, v

def learn_network_adam(raw_data, method, train_dat):

    num_spin, num_round = train_dat['cur_h'].shape
    
    num_epoch, stepsz, rec_gap = (train_dat.get(key, None) for key in ["num_epoch", "stepsz", "rec_gap"])
    list_step = np.arange(num_epoch, 0, - rec_gap)[::-1]
    
    cur_j, cur_h =  (train_dat.get(key, None) for key in ["cur_j", "cur_h"])

    save_path = train_dat.get('save_path', None)

    backtrack_gap, backtrack_tol = (train_dat.get(key, None) for key in ["backtrack_gap", "backtrack_tol"])
    backtrack_counter = 0

    rec_jmat_all = np.zeros((num_epoch, num_spin, num_spin))
    rec_hvec_all = np.zeros((num_epoch, num_spin, num_round))
    rec_jgrad_sum_norm = np.inf * np.ones(num_epoch) 

    mjj, vjj = np.zeros(cur_j.shape), np.zeros(cur_j.shape)
    mhh, vhh = np.zeros(cur_h.shape), np.zeros(cur_h.shape)

    log_adam_grad = {name: deque(maxlen=backtrack_gap) for name in ["mjj", "vjj", "mhh", "vhh"]}

    counter = 1
    while counter <= num_epoch:

        rec_jgrad, rec_hgrad = compute_gradient(cur_j, cur_h, raw_data, method, train_dat)
        
        rec_jgrad, rec_hgrad = apply_regularization(rec_jgrad, rec_hgrad, cur_j, cur_h, train_dat)

        # Adam updates
        rec_jgrad_sum = np.sum(rec_jgrad, axis=2)
        update, mjj, vjj = update_adam(rec_jgrad_sum, mjj, vjj, counter, stepsz)
        cur_j -= update

        update, mhh, vhh = update_adam(rec_hgrad, mhh, vhh, counter, stepsz)
        cur_h -= update

        rec_jmat_all[counter - 1, :, :] = cur_j
        rec_hvec_all[counter - 1, :, :] = cur_h
        rec_jgrad_sum_norm[counter - 1] = np.linalg.norm(rec_jgrad_sum)

        for name, value in zip(["mjj", "vjj", "mhh", "vhh"], [mjj, vjj, mhh, vhh]):
            log_adam_grad[name].append(value)

        if counter in list_step:
            
            sio.savemat(save_path + 'train_log.mat', {
                'list_step':list_step, 'rec_hvec_all':rec_hvec_all, 'rec_jmat_all':rec_jmat_all, 'rec_jgrad_sum_norm':rec_jgrad_sum_norm})
                                    
            print('Progress: %d, Network gradient: %f' %(np.round(100 * counter / num_epoch, 2), rec_jgrad_sum_norm[counter - 1]))


        if counter > backtrack_gap and rec_jgrad_sum_norm[counter - 1] > 2 * rec_jgrad_sum_norm[counter - 1 - backtrack_gap]:
            print('Backtracking at epoch %d' % counter)
            backtrack_counter += 1
            mjj, vjj, mhh, vhh = [log_adam_grad[key][0] for key in ['mjj', 'vjj', 'mhh', 'vhh']]
            counter = counter - backtrack_gap
            stepsz = stepsz / 4
            if backtrack_counter > backtrack_tol:
                print('Backtracking more than %d times, stop training.' % backtrack_tol)
                break
        else:
            counter += 1

    pos = np.argmin(rec_jgrad_sum_norm)
    cur_h = rec_hvec_all[pos, :, :]
    cur_j = rec_jmat_all[pos, :, :]

    return cur_j, cur_h