from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
import anndata
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


from util.compute import (
    compute_onmf,
    summarize_onmf_decomposition,
    learn_jmat_adam,
    learn_jmat_adam2,
    prepare_onmf_decomposition,
    select_diverse_sample,
    onmf_discretize,
    sample_corr_mean,
    preprocess_sampling
)

from util.plotting import (
    onmf_to_csv, 
    gene_program_decomposition, 
    temporary_spin_name
)


class AbstractDSPIN(ABC):
    def __init__(self, 
                 adata: anndata.AnnData,
                 save_path: str,
                 num_spin: int = 10,
                 num_onmf_components: int = None,
                 num_repeat: int = 10,
                 filter_threshold: float = 0.02):
        # Filter out low expressed genes
        self.filter_threshold = filter_threshold
        counts_threshold = int(adata.shape[0] * self.filter_threshold)
        sc.pp.filter_genes(adata, min_counts=counts_threshold)

        self.adata = adata
        self.save_path = save_path
        self.num_spin = num_spin
        self.num_repeat = num_repeat
        
        if num_onmf_components is None:
            self.num_onmf_components = num_spin
        else:
            self.num_onmf_components = num_onmf_components

        if self.num_spin > 10:
            warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

        if self.num_spin > self.num_onmf_components:
            raise ValueError("num_spin must be less than or equal to num_onmf_components.")
        
        if not os.path.exists(self.save_path):
            raise ValueError("save_path does not exist.")
        
        self._network = None
        self._responses = None
        self._onmf_rep_ori = None
        self._onmf_rep_tri = None
        self._raw_data = None
    
    # To add more restrictions/ check on the attributes
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def discretize(self) -> np.ndarray:
        """
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.

        Returns:
        - np.ndarray: The discretized ONMF representation.
        """
        onmf_rep_ori = self.onmf_rep_ori
        fig_folder = self.save_path + 'figs/'
        
        onmf_rep_tri = onmf_discretize(onmf_rep_ori, fig_folder)        
        self.onmf_rep_tri = onmf_rep_tri
    
    def cross_corr(self, sample_col_name) -> np.ndarray:
        adata = self.adata
        onmf_rep_tri = self.onmf_rep_tri
        save_path = self.save_path
        raw_data, samp_list = sample_corr_mean(adata.obs[sample_col_name], onmf_rep_tri)
        self.raw_data = raw_data
        self.samp_list = samp_list
        
        filename = f"{save_path}data_raw.mat"
        savemat(filename, {'raw_data': raw_data, 'network_subset': list(range(len(samp_list))), 'samp_list': samp_list})
    
    def common_hyperparas_setting(self, example_list=None) -> dict:
        raw_data = self.raw_data
        num_spin = self.num_spin

        if example_list:
            example_list_ind = [list(self.samp_list).index(samp) for samp in example_list]
            raw_data = raw_data[example_list_ind]

        num_spin = raw_data[0][0].shape[0]
        num_samp = len(raw_data)
        rec_all_corr = np.zeros((num_spin, num_spin, num_samp))
        rec_all_mean = np.zeros((num_spin, num_samp))

        for ii in range(num_samp):
            rec_all_corr[:, :, ii] = raw_data[ii][0]
            rec_all_mean[:, ii] = raw_data[ii][1].flatten()
        
        cur_j = np.zeros((num_spin, num_spin))
        cur_h = np.zeros((num_spin, num_samp))
        data_dir = self.save_path + 'dspin_python/'
        task_name = data_dir + 'train_log'

        train_dat = {'cur_j': cur_j, 'cur_h': cur_h, 'task_name': task_name,
                     'rec_all_corr': rec_all_corr, 'rec_all_mean': rec_all_mean}

        dir_list = [data_dir, data_dir + 'train_log']
        for directory in dir_list:
            if not os.path.exists(directory):
                os.makedirs(directory)

        return train_dat

    @abstractmethod
    def network_construct(self, specific_hyperparams: dict):
        pass

    @abstractmethod
    def network_infer(self, sample_col_name: str):
        pass



class SmallDSPIN(AbstractDSPIN):
    def __init__(self, 
                 adata: anndata.AnnData,
                 save_path: str,
                 num_spin: int = 10,
                 num_onmf_components: int = None,
                 num_repeat: int = 10):
        super().__init__(adata, save_path, num_spin, num_onmf_components, num_repeat)
        print("SmallDSPIN initialized.")
        self._onmf_rep_ori = adata.X


    @property
    def onmf_rep_ori(self):
        return self.adata.X
    
    def network_construct(self, 
                          specific_hyperparams: 
                          dict = {'epoch': 200, 'spin_thres': 16,
                                  'stepsz': 0.2, 'dropout': 0, 'counter': 1,
                                  'samplingsz': 5e7, 'samplingmix': 1e3, 'rec_gap': 10}):

        train_dat = self.common_hyperparas_setting()
        train_dat.update(specific_hyperparams)

        rec_all_corr = train_dat['rec_all_corr']
        rec_all_mean = train_dat['rec_all_mean']
        cur_j, cur_h = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat)
        self._network = cur_j
        self._responses = cur_h

    def network_infer(self, sample_col_name: str):
        self.discretize()
        self.cross_corr(sample_col_name)
        self.network_construct()



class LargeDSPIN(AbstractDSPIN):
    def __init__(self, 
                    adata: anndata.AnnData,
                    save_path: str,
                    num_spin: int = 10,
                    num_onmf_components: int = None,
                    preprogram: list = None,
                    num_repeat: int = 10,
                    filter_threshold: float = 0.02):
            super().__init__(adata, save_path, num_spin, num_onmf_components, num_repeat, filter_threshold)
            print("LargeDSPIN initialized.")
            self._onmf_summary = None
            self._gene_matrix_large = None
            self._use_data_list = None
            self._gene_program_csv = None
            self._preprogram = None
    
    @property
    def optimized_algorithm(self):
        if self.num_spin < 25:
            return 1
        else:
            return 2
    
    @optimized_algorithm.setter
    def optimized_algorithm(self, value):
        self._optimized_algorithm = value
    
    @property
    def specific_hyperparams(self):
        if self.optimized_algorithm == 1:
            return {'epoch': 150, 'spin_thres': 16,
                    'stepsz': 0.02, 'dropout': 0, 'counter': 1,
                    'samplingsz': 5e6, 'samplingmix': 1e3, 'rec_gap': 10,
                    'lam_12h': 0.005, 'lam_l1j': 0.01}
        elif self.optimized_algorithm == 2:
            return {'decay_rate': 0.001, 'num_iterations': 400,
                    'initial_learning_rate': 0.05, 
                    'l2_h': 5e7, 'lam_l1j': 0.005, 'l1_thres': 0.02}

    @specific_hyperparams.setter
    def specific_hyperparams(self, value):
        self._specific_hyperparams = value

    @property
    def example_list(self):
        return self._example_list
    
    @example_list.setter
    def example_list(self, value):
        self._example_list = value
    
    def __setattr__(self, name, value):
        if name == 'example_list':
            self._example_list = value
        elif name == 'specific_hyperparams':
            self._specific_hyperparams = value
        elif name == 'optimized_algorithm':
            self._optimized_algorithm = value
        elif name == 'preprogram':
            self._preprogram = value
        else:
            return super().__setattr__(name, value)
    
    def __getattr__(self, name):
        return super().__getattr__(name)

    def matrix_balance(self):
        std, matrix_path = prepare_onmf_decomposition(self.adata, self.save_path)
        cur_matrix = np.load(matrix_path)
        self.gene_matrix_large = cur_matrix
        self.matrix_std = std

    def onmf_abstract(self, balance_by='leiden', total_sample_size=2e4, method='squareroot'):
        #TODO: this huge function needs to be factored out
        adata = self.adata
        preprogram = self.preprogram
        
        sampling_number, cluster_list = preprocess_sampling(adata, balance_by, total_sample_size, method, maximum_sample_rate=2)
        
        gene_matrix_balanced = np.zeros((np.sum(sampling_number), adata.X.shape[1]))

        # Pre-computing num_repeat times
        print("Pre-computing")
        for seed in range(1, self.num_repeat + 1):

            print(f"Round_{seed}")
            np.random.seed(seed) 
            for ii in range(len(cluster_list)):
                cur_num = sampling_number[ii]
                cur_filt = adata.obs[balance_by] == cluster_list[ii]
                sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
                strart_ind = np.sum(sampling_number[:ii])
                end_ind = strart_ind + cur_num
                gene_matrix_balanced[strart_ind: end_ind, :] = adata.X[cur_filt, :][sele_ind, :]
                std = gene_matrix_balanced.std(axis=0)
                gene_matrix_balanced /= std.clip(np.percentile(std, 20), np.inf)
                matrix_path = self.save_path + 'gmatrix_' + '{:.0e}'.format(total_sample_size) + '_balanced_' + method + '_' + str(seed) + '.npy'
                np.save(matrix_path, gene_matrix_balanced)

            current_onmf = compute_onmf(seed, self.num_spin, gene_matrix_balanced)
            np.save(f"{self.save_path}onmf_{self.num_spin}_{seed}.npy", current_onmf)

        # Summarizing the ONMF result
        onmf_summary = summarize_onmf_decomposition(self.num_spin, self.num_repeat, self.num_onmf_components, 
                                                    onmf_path= self.save_path, 
                                                    gene_matrix = self.gene_matrix_large,
                                                    fig_folder= self.save_path )
        np.save(f"{self.save_path}onmf_summary_{self.num_spin}.npy", onmf_summary)
        self.onmf_summary = onmf_summary

        # Save ONMF summary to CSV
        features = onmf_summary.components_
        gene_names = self.adata.var_names
        gene_program_filename = onmf_to_csv(features, gene_names, self.save_path, thres=0.05)
        self.gene_program_csv = gene_program_filename

    def gene_program_discovery(self, num_gene_select: int = 10, n_clusters: int = 4, sample_column_name = None, **kwargs):
        #TODO: kwargs needed to be change later because it is not friendly for users
        self.matrix_balance()
        self.onmf_abstract(**kwargs)
        self.compute_onmf_rep_ori()

        print('Discretize ONMF representation into three states')
        self.discretize()
        self.cross_corr(sample_column_name)
        spin_name = temporary_spin_name(self.gene_program_csv)
        gene_program_decomposition(self.onmf_summary,
                                   self.num_spin,
                                   spin_name,
                                   self.adata.X.astype(np.float64),
                                   self.onmf_rep_tri,
                                   self.save_path + 'figs/',
                                   num_gene_select, 
                                   n_clusters)
    
    def compute_onmf_rep_ori(self) -> np.ndarray:
        """
        Computes the original ONMF representation given the ONMF summary.
        """

        gene_matrix = self.adata.X.astype(np.float64)
        gene_matrix /= self.matrix_std
        onmf_rep_ori = self.onmf_summary.transform(gene_matrix)
        self.onmf_rep_ori = onmf_rep_ori
    
    def post_processing(self):
        # balance the experimental conditions by clustering and downsampling
        raw_data = self.raw_data
        fig_folder = self.save_path + 'figs/'
        use_data_list = select_diverse_sample(raw_data, num_cluster=32, fig_folder=fig_folder)
        self.use_data_list = use_data_list
    
    def network_construct(self, specific_hyperparams=None):
        train_dat = self.common_hyperparas_setting(self.example_list)
        train_dat.update(self.specific_hyperparams)

        rec_all_corr = train_dat['rec_all_corr']
        rec_all_mean = train_dat['rec_all_mean']
        if self.optimized_algorithm == 1:
            cur_j, cur_h = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat)
        elif self.optimized_algorithm == 2:
            cur_j, cur_h = learn_jmat_adam2(rec_all_corr, rec_all_mean, train_dat)
        
        self._network = cur_j
        self._responses = cur_h

    def network_infer(self, example_list):
        self.example_list = example_list
        self.post_processing()
        self.network_construct()

# Select the class to use
class DSPIN(object):
    def __new__(cls, *args, **kwargs):
        if kwargs['num_spin'] < 15:
            return SmallDSPIN(*args, **kwargs)
        else:
            return LargeDSPIN(*args, **kwargs)