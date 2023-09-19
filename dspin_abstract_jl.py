# -*-coding:utf-8 -*-
'''
@Time    :   2023/09/14 15:43
@Author  :   Jialong Jiang, Yingying Gong
'''

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
import anndata as ad
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from tqdm import tqdm
from scipy.io import savemat, loadmat
from scipy.sparse import issparse
import networkx as nx
import matplotlib.patheffects as patheffects
import warnings
import itertools
from typing import List

'''
from util.compute import (
    compute_onmf,
    summarize_onmf_decomposition,
    learn_jmat_adam,
    learn_jmat_adam2,
    prepare_onmf_decomposition,
    select_diverse_sample,
    onmf_discretize,
    sample_corr_mean,
    preprocess_sampling,
    balanced_gene_matrix
)

from util.plotting import (
    onmf_to_csv, 
    gene_program_decomposition, 
    temporary_spin_name
)


'''

from util.compute_new import (
    onmf_discretize,
    sample_corr_mean,
    learn_network_adam,
    category_balance_number,
    compute_onmf,
    summary_components
)

from util.plot_new import (
    onmf_to_csv
)

class AbstractDSPIN(ABC):
    def __init__(self, 
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        
        self.adata = adata
        self.save_path = os.path.abspath(save_path) + '/'
        self.num_spin = num_spin

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("Saving path does not exist. Creating a new folder.")
        self.fig_folder = self.save_path + 'figs/'
        os.makedirs(self.fig_folder, exist_ok=True)


    @property
    def onmf_rep_ori(self):
        return self._onmf_rep_ori
    
    @property
    def onmf_rep_tri(self):
        return self._onmf_rep_tri
    
    @property
    def raw_data(self):
        return self._raw_data

    @property
    def network(self):
        return self._network

    @property
    def responses(self):
        return self._responses
    
    @onmf_rep_ori.setter
    def onmf_rep_ori(self, value):
        self._onmf_rep_ori = value
    
    @network.setter
    def network(self, value):
        self._network = value
    
    @responses.setter
    def responses(self, value):
        self._responses = value

    def discretize(self) -> np.ndarray:
        """
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.

        Returns:
        - np.ndarray: The discretized ONMF representation.
        """
        onmf_rep_ori = self.onmf_rep_ori
        fig_folder = self.fig_folder
        
        onmf_rep_tri = onmf_discretize(onmf_rep_ori, fig_folder)        
        self._onmf_rep_tri = onmf_rep_tri

    def raw_data_corr(self, sample_col_name) -> np.ndarray:

        raw_data, samp_list = sample_corr_mean(self.adata.obs[sample_col_name], self.onmf_rep_tri)
        self._raw_data = raw_data
        self.samp_list = samp_list

    def raw_data_state(self, sample_col_name) -> np.ndarray:

        cadata = self.adata
        onmf_rep_tri = self.onmf_rep_tri   

        samp_list = np.unique(cadata.obs[sample_col_name])
        state_list = np.zeros(len(samp_list), dtype=object)

        for ii, cur_samp in enumerate(samp_list):
            cur_filt = cadata.obs[sample_col_name] == cur_samp
            cur_state = self.onmf_rep_tri[cur_filt, :]
            state_list[ii] = cur_state.T
        
        self._raw_data = state_list
        self.samp_list = samp_list

    def default_params(self, 
                       method: str) -> dict:
        
        num_spin = self.num_spin
        raw_data = self.raw_data

        if self.example_list is not None:
            example_list_ind = [list(self.samp_list).index(samp) for samp in self.example_list]
            raw_data = raw_data[example_list_ind]
        
        num_sample = len(raw_data)
        params = {'num_epoch': 200, 
                  'cur_j': np.zeros((num_spin, num_spin)),
                  'cur_h': np.zeros((num_spin, num_sample)),
                  'save_path': self.save_path}
        params.update({'lambda_l1_j': 0.01,
                       'lambda_l1_h': 0,
                       'lambda_l2_j': 0,
                       'lambda_l2_h': 0.05,
                       'lambda_prior_h': 0})
        params.update({'backtrack_gap': 20,
                       'backtrack_tol': 4})

        if method == 'maximum_likelihood':
            params['stepsz'] = 0.1
        elif method == 'mcmc_maximum_likelihood':
            params['stepsz'] = 0.01
            params['mcmc_samplingsz'] = 2e5
            params['mcmc_samplingmix'] = 1e3
            params['mcmc_samplegap'] = 1
        else:
            params['stepsz'] = 0.05

        return params
        

    def network_infer(self, 
                      sample_col_name: str,
                      method: str = 'auto',
                      params: dict = None,
                      example_list: List[str] = None,
                      record_step: int = 10):
        
        if method not in ['maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood', 'auto']:
            raise ValueError("Method must be one of 'maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood', or 'auto'.")

        if method == 'auto':
            if self.num_spin <= 12:
                method = 'maximum_likelihood'
            elif self.num_spin <= 25:
                method = 'mcmc_maximum_likelihood'
            else:
                method = 'pseudo_likelihood'

        print("Using {} for network inference.".format(method))

        self.discretize()

        if method == 'pseudo_likelihood':
            self.raw_data_state(sample_col_name)
        else: 
            self.raw_data_corr(sample_col_name)

        self.example_list = example_list

        train_dat = self.default_params(method)
        train_dat['rec_gap'] = record_step
        if params is not None:
            train_dat.update(params)

        cur_j, cur_h = learn_network_adam(self.raw_data, method, train_dat)
        self._network = cur_j
        self._responses = cur_h

class SmallDSPIN(AbstractDSPIN):
    def __init__(self, 
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        super().__init__(adata, save_path, num_spin)
        print("SmallDSPIN initialized.")
        self._onmf_rep_ori = adata.X

class LargeDSPIN(AbstractDSPIN):
    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int,
                 num_onmf_components: int = None,
                 preprograms: List[List[str]] = None,
                 num_repeat: int = 10):
            super().__init__(adata, save_path, num_spin)

            print("LargeDSPIN initialized.")

            self._onmf_summary = None
            self._gene_matrix_large = None
            self._use_data_list = None
            self.gene_program_csv = None
            self.preprograms = None
            self.preprogram_num = len(preprograms) if preprograms else 0

    def subsample_matrix_balance(self, 
                                 total_sample_size: int, 
                                 std_clip_percentile: float = 20) -> (np.ndarray, np.ndarray):
        
        gene_matrix = self.adata.X
        cadata = self.adata
        num_cell, num_gene = gene_matrix.shape

        balance_by = self.balance_obs
        method = self.balance_method
        maximum_sample_rate = self.max_sample_rate

        if method == None:
            gene_matrix_balanced = gene_matrix[np.random.choice(num_cell, total_sample_size, replace=False), :]
        else:
            cluster_list = list(cadata.obs[balance_by].value_counts().keys())
            cluster_count = list(cadata.obs[balance_by].value_counts())

            sampling_number = category_balance_number(total_sample_size, cluster_count, method, maximum_sample_rate)

            gene_matrix_balanced = np.zeros((np.sum(sampling_number), cadata.X.shape[1]))
    
            for ii in range(len(cluster_list)):
                cur_num = sampling_number[ii]
                cur_filt = cadata.obs[balance_by] == cluster_list[ii]
                sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
                strart_ind = np.sum(sampling_number[:ii])
                end_ind = strart_ind + cur_num
                if issparse(cadata.X):
                    gene_matrix_balanced[strart_ind: end_ind, :] = cadata.X[cur_filt, :][sele_ind, :].toarray()
                else:
                    gene_matrix_balanced[strart_ind: end_ind, :] = cadata.X[cur_filt, :][sele_ind, :]

        std = gene_matrix_balanced.std(axis=0)
        std_clipped = std.clip(np.percentile(std, std_clip_percentile), np.inf)
        gene_matrix_balanced_normalized = gene_matrix_balanced / std_clipped

        return std_clipped, gene_matrix_balanced_normalized
                                 
    def compute_onmf_repeats(self, 
                             num_onmf_components: int,
                             num_repeat: int,
                             num_subsample: int,
                             seed: int = 0,
                             std_clip_percentile: float = 20):

        preprograms = self.prior_programs
        adata = self.adata
        os.makedirs(self.save_path + 'onmf/', exist_ok=True)

        print("Computing ONMF decomposition...")
        
        for ii in range(num_repeat):
            np.random.seed(seed + ii)
            if os.path.exists(self.save_path + 'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii)):
                print("ONMF decomposition {} already exists. Skipping...".format(ii))
                continue
            _, gene_matrix_norm = self.subsample_matrix_balance(num_subsample, std_clip_percentile=std_clip_percentile)

            current_onmf = compute_onmf(seed + ii, num_onmf_components, gene_matrix_norm[:, self.prior_programs_mask])
            np.save(self.save_path + 'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii), current_onmf)

    def summarize_onmf_result(self, 
                              num_onmf_components: int,
                              num_repeat: int,
                              summary_method: str):
        num_spin = self.num_spin
        adata = self.adata
        prior_programs_mask = self.prior_programs_mask

        rec_components = np.zeros((num_repeat, num_onmf_components, np.sum(prior_programs_mask)))

        for ii in range(num_repeat):
            cur_onmf = np.load(self.save_path + 'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii), allow_pickle=True).item()
            rec_components[ii] = cur_onmf.components_

        all_components = rec_components.reshape(-1, np.sum(prior_programs_mask))

        gene_group_ind = summary_components(all_components, num_onmf_components, summary_method=summary_method)

        sub_mask_ind = np.where(prior_programs_mask)[0]
        gene_group_ind = [sub_mask_ind[gene_list] for gene_list in gene_group_ind]
        gene_group_ind += self.prior_programs_ind

        components_summary = np.zeros((num_spin, adata.shape[1]))
        for ii in range(num_spin):
            sub_matrix = self.large_subsample_matrix[:, gene_group_ind[ii]]
            sub_onmf = NMF(n_components=1, init='random', random_state=0).fit(sub_matrix)
            components_summary[ii, gene_group_ind[ii]] = sub_onmf.components_

        components_summary = normalize(components_summary, axis=1, norm='l2')
        onmf_summary = NMF(n_components=num_spin, init='random', random_state=0)
        onmf_summary.components_ = components_summary

        return onmf_summary

    def gene_program_discovery(self, 
                               num_onmf_components: int = None,
                               num_subsample: int = 10000,
                               num_subsample_large: int = None,
                               num_repeat: int = 10,
                               balance_obs: str = None,
                               balance_method: str = None,
                               max_sample_rate: float = 2,
                               prior_programs: List[List[str]] = None, 
                               summary_method: str = 'kmeans',):
        
        if balance_method not in ['equal', 'proportional', 'squareroot', None]:
            raise ValueError('balance_method must be one of equal, proportional, squareroot, or None')
        
        if summary_method not in ['kmeans', 'leiden']:
            raise ValueError('summary_method must be one of kmeans or leiden')
               
        adata = self.adata

        if num_subsample_large is None:
            num_subsample_large = min(num_subsample * 5, self.adata.shape[0])

        if (balance_obs is not None) & (balance_method is None):
            balance_method = 'squareroot'

        self.balance_obs = balance_obs
        self.balance_method = balance_method
        self.max_sample_rate = max_sample_rate
        self.prior_programs = prior_programs

        if prior_programs is not None:
            prior_program_ind = [
                [np.where(adata.var_names == gene)[0][0] for gene in gene_list]
                for gene_list in prior_programs]
            preprogram_flat = [gene for program in prior_programs for gene in program]
            if len(prior_programs) > num_spin:
                raise ValueError('Number of preprograms must be less than the number of spins')
            prior_program_mask = ~ np.isin(adata.var_names, preprogram_flat)
        else:
            prior_program_mask = np.ones(adata.shape[1], dtype=bool)
            prior_program_ind = []

        if num_onmf_components is None:
            num_onmf_components = self.num_spin - len(prior_program_ind)

        self.prior_programs_mask = prior_program_mask
        self.prior_programs_ind = prior_program_ind

        self.matrix_std, self.large_subsample_matrix = self.subsample_matrix_balance(num_subsample_large, std_clip_percentile=20)

        self.compute_onmf_repeats(num_onmf_components, num_repeat, num_subsample, std_clip_percentile=20)

        onmf_summary = self.summarize_onmf_result(num_onmf_components, num_repeat, summary_method)
        self._onmf_summary = onmf_summary

        onmf_to_csv(onmf_summary.components_, adata.var_names, self.save_path + 'gene_programs_{}_{}_{}.csv'.format(len(prior_program_ind), num_onmf_components, num_spin), thres=0.01)

        gene_matrix = self.adata.X
        if issparse(gene_matrix):
            gene_matrix = np.asarray(gene_matrix.toarray()).astype(np.float64)
        self._onmf_rep_ori = onmf_summary.transform(gene_matrix / self.matrix_std)



        


                
class DSPIN(object):
    def __new__(cls, 
                adata: ad.AnnData,
                save_path: str,
                num_spin: int = 15,
                filter_threshold: float = 0.02, 
                **kwargs):
        # Note: when initializing an object, please set 'num_spin' = the desired value in the arguments

        sc.pp.filter_genes(adata, min_cells=adata.shape[0] * filter_threshold)
        print('{} genes have expression in more than {} of the cells'.format(adata.shape[1], filter_threshold))

        if adata.shape[1] <= num_spin:
            return SmallDSPIN(adata, save_path, num_spin=num_spin, **kwargs)
        else:
            return LargeDSPIN(adata, save_path, num_spin=num_spin, **kwargs)
        

if __name__ == "__main__":


    data_folder = 'data/thomsonlab_signaling/'
    large_data_folder = 'large_data/thomsonlab_signaling/'

    ''' 
    cadata = ad.read_h5ad(large_data_folder + 'thomsonlab_signaling_filtered_2500_scvi_umap.h5ad')

    random_select = np.random.choice(cadata.shape[0], 10000, replace=False)
    gene_select = np.random.choice(cadata.shape[1], 500, replace=False)
    cadata = cadata[random_select, :][:, gene_select].copy()
    cadata.write(large_data_folder + 'thomsonlab_signaling_filtered_2500_scvi_umap_small.h5ad')
    
    '''
    cadata = ad.read_h5ad(large_data_folder + 'thomsonlab_signaling_filtered_2500_scvi_umap_small.h5ad')

    save_path = "test/test_signalling_0913/"

    samp_list = np.unique(cadata.obs['sample_id'])
    subset_list = samp_list[: 3]

    num_spin = 10
    num_spin = 30
    num_spin = 20
    model = DSPIN(cadata, save_path, num_spin=num_spin)

    # prior_programs = ['CD3D PTPRCAP IL7R LCK PRDX2 ETS1 S1PR4'.split(' '), 'ACTB FTH1 CYBA'.split(' '), ['IRF1']] 

    # model.gene_program_discovery(balance_obs='leiden', prior_programs=prior_programs)
    model.gene_program_discovery(balance_obs='leiden')
    model.network_infer(sample_col_name='sample_id', example_list=subset_list)
    
    ''' 
    data_folder = 'data/HSC_simulation/'

    cadata = ad.read_h5ad(data_folder + 'hsc_simulation_with_perturbations_small.h5ad')

    # random_select = np.random.choice(cadata.shape[0], 10000, replace=False)
    # cadata = cadata[random_select, :].copy()

    # cadata.write(data_folder + 'hsc_simulation_with_perturbations_small.h5ad')

    save_path = 'test/hsc_test0912'

    samp_list = np.unique(cadata.obs['sample_id'])
    subset_list = samp_list[: 3]

    num_spin = cadata.shape[1]
    model = DSPIN(cadata, save_path, num_spin=num_spin)

    model.network_infer(sample_col_name='sample_id', example_list=subset_list)
'''
