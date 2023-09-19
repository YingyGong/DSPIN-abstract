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
from tqdm import tqdm


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

from sklearn.preprocessing import normalize

def summary_components(all_components, num_spin, summary_method='kmeans'):
    
    num_gene = all_components.shape[1]

    if summary_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_spin, random_state=0, n_init=50).fit(all_components)
        kmeans_gene = KMeans(n_clusters=2 * num_spin, random_state=0, n_init=10).fit(all_components.T)

        components_kmeans = np.zeros((num_spin, num_gene))
        for ii in range(num_spin):
            components_kmeans[ii] = np.mean(all_components[kmeans.labels_ == ii], axis=0)
        components_kmeans = normalize(components_kmeans, axis=1, norm='l2')

        gene_groups_ind = []
        for ii in range(num_spin):
            gene_groups_ind.append(np.argmax(components_kmeans, axis=0) == ii)

    return gene_groups_ind
    
from scipy.linalg import orth 
from sklearn.decomposition import NMF

def onmf(X, rank, max_iter=100):
    
    m, n = X.shape
    
    A = np.random.rand(m, rank) 
    S = np.random.rand(rank, n)
    S = np.abs(orth(S.T).T)
    
    pbar = tqdm(total=max_iter, desc = "Iteration Progress")

    for itr in range(max_iter):
            
        coef_A = X.dot(S.T) / A.dot(S.dot(S.T))
        A = np.nan_to_num(A * coef_A) #, posinf=1e5)

        AtX = A.T.dot(X)
        coef_S = AtX / S.dot(AtX.T).dot(S)
        S = np.nan_to_num(S * coef_S) #, posinf=1e5)

        pbar.update(1)

        if itr % 10 == 0:
            error = np.linalg.norm(X - np.dot(A, S), 'fro')
            pbar.set_postfix({"Reconstruction Error": f"{error:.2f}"})

    
    pbar.close()

    norm_fac = np.sqrt(np.diag(S.dot(S.T)))
    S = S / norm_fac.reshape(- 1, 1)
    A = A * norm_fac.reshape(1, - 1)
    
    as_prod = A.dot(S)
    const = np.sum(X * as_prod) / np.sum(as_prod ** 2)
    A *= const

    return S.T, A

def compute_onmf(seed, num_spin, gene_matrix_bin):
    np.random.seed(seed)
    H, W = onmf(gene_matrix_bin, num_spin)

    nmf_model = NMF(n_components=num_spin, random_state=seed)
    nmf_model.components_ = np.array(H).T
    nmf_model.n_components_ = num_spin
    return nmf_model

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

import numba

@numba.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@numba.njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@numba.jit()
def pseudol_gradient(cur_j, cur_h, cur_state):
    num_spin = cur_j.shape[0]

    cur_j_grad = np.zeros((num_spin, num_spin))
    cur_h_grad = np.zeros((num_spin, 1))

    j_filt = cur_j.copy()
    np.fill_diagonal(j_filt, 0)
    effective_h = j_filt.dot(cur_state) + cur_h

    for ii in range(num_spin):
        j_sub = cur_j[ii, ii]
        h_sub = effective_h[ii, :]

        term1 = np.exp(j_sub + h_sub)
        term2 = np.exp(j_sub - h_sub)

        j_sub_grad = cur_state[ii, :] ** 2 - (term1 + term2) / (term1 + term2 + 1)
        h_eff_grad = cur_state[ii, :] - (term1 - term2) / (term1 + term2 + 1)

        j_off_sub_grad = h_eff_grad * cur_state

        cur_j_grad[ii, :] = np_mean(j_off_sub_grad, axis=1)
        cur_j_grad[ii, ii] = np.mean(j_sub_grad)

        cur_h_grad[ii] = np.mean(h_eff_grad)

        cur_j_grad = (cur_j_grad + cur_j_grad.T) / 2

    return - cur_j_grad, - cur_h_grad

@numba.njit()
def samp_moments(j_mat, h_vec, sample_size, mixing_time, samp_gap):
    per_batch = int(1e5)
    
    num_spin = j_mat.shape[0]
    rec_corr = np.zeros((num_spin, num_spin))
    rec_mean = np.zeros(num_spin)
    beta = 1
    batch_count = 1
    rec_sample = np.empty((num_spin, min(per_batch, int(sample_size))))
    cur_spin = (np.random.randint(0, 3, (num_spin, 1)) - 1).astype(np.float64)
    tot_sampling = int(mixing_time + sample_size * samp_gap - mixing_time % samp_gap)

    rand_ind = np.random.randint(0, num_spin, tot_sampling)
    rand_flip = np.random.randint(0, 2, tot_sampling)
    rand_prob = np.random.rand(tot_sampling)

    # for ii in numba.prange(tot_sampling):
    for ii in range(tot_sampling):
        cur_ind = rand_ind[ii]
        j_sub = j_mat[cur_ind, :]
        accept_prob = 0.0
        new_spin = 0.0
        diff_energy = 0.0

        if cur_spin[cur_ind] == 0:
            if rand_flip[ii] == 0:
                new_spin = 1.0
            else:
                new_spin = -1.0
            diff_energy = -j_mat[cur_ind, cur_ind] - new_spin * (j_sub.dot(cur_spin) + h_vec[cur_ind])
            accept_prob = min(1.0, np.exp(- diff_energy * beta)[0])
        else:
            if rand_flip[ii] == 0:
                accept_prob = 0;
            else:
                diff_energy = cur_spin[cur_ind] * (j_sub.dot(cur_spin) + h_vec[cur_ind])
                accept_prob = min(1.0, np.exp(- diff_energy * beta)[0])

        if rand_prob[ii] < accept_prob:
            if cur_spin[cur_ind] == 0:
                cur_spin[cur_ind] = new_spin
            else:
                cur_spin[cur_ind] = 0

        if ii > mixing_time:
            if (ii - mixing_time) % samp_gap == 0:
                rec_sample[:, batch_count - 1] = cur_spin[:, 0].copy()
                batch_count += 1

                if batch_count == per_batch + 1:
                    batch_count = 1
                    rec_sample = np.ascontiguousarray(rec_sample)
                    rec_corr += rec_sample.dot(rec_sample.T)
                    rec_mean += np.sum(rec_sample, axis=1)

    if batch_count != 1:
        cur_sample = rec_sample[:, :batch_count - 1]
        cur_sample = np.ascontiguousarray(cur_sample)
        rec_corr += cur_sample.dot(cur_sample.T)
        rec_mean += np.sum(cur_sample, axis=1)

    corr_para = rec_corr / sample_size
    mean_para = rec_mean / sample_size

    return corr_para, mean_para

from collections import deque

def compute_gradient(cur_j, cur_h, raw_data, method, train_dat):

    num_spin, num_round = cur_h.shape

    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))
    
    for kk in range(num_round):

        if method == 'pseudo_likelihood':
            j_grad, h_grad = pseudol_gradient(cur_j, cur_h[:, kk].reshape(- 1, 1), raw_data[kk])
            h_grad = h_grad.flatten()
        else:
            if method == 'maximum_likelihood':
                corr_para, mean_para = para_moments(cur_j, cur_h[:, kk])            
            elif method == 'mcmc_maximum_likelihood':
                corr_para, mean_para = samp_moments(cur_j, cur_h, train_dat['mcmc_samplingsz'], train_dat['mcmc_samplingmix'], train_dat['mcmc_samplegap'])
            j_grad = corr_para - raw_data[kk][0]
            h_grad = mean_para - raw_data[kk][1].flatten()

        rec_jgrad[:, :, kk] = j_grad
        rec_hgrad[:, kk] = h_grad

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