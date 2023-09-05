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
    corr_mean,
    learn_jmat_adam,
    prepare_onmf_decomposition,
    select_diverse_sample,
    onmf_discretize
)

from util.plotting import onmf_to_csv, gene_program_decomposition, temporary_spin_name

def sample_corr_mean(samp_full, comp_bin):
    
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)
    
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])
        
    return raw_corr_data, samp_list

class DSPIN:
    def __init__(self,
                 adata: anndata.AnnData,
                 save_path: str,
                 num_spin: int = 10,
                 num_onmf_components: int = None,
                 num_repeat: int = 10,
                 epoch: int = 150,
                 spin_thres: int = 16,
                 stepsz: float = 0.02,
                 dropout: int = 0,
                 counter: int = 1,
                 samplingsz: float = 5e6,
                 samplingmix: float = 1e3,
                 rec_gap: int = 10,
                 lam_12h: float = 0.005,
                 lam_l1j: float = 0.01):
        self.adata = adata
        self.save_path = save_path
        
        self.num_spin = num_spin
        
        if num_onmf_components is None:
            self.num_onmf_components = num_spin
        else:
            self.num_onmf_components = num_onmf_components
        self.num_repeat = num_repeat

        num_gene = self.adata.X.shape[1]
        if num_gene > num_spin:
            self._onmf_indicator = True
        else:
            self._onmf_indicator = False
            self._onmf_rep_ori = adata.X

        if self.num_spin > 10:
            warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

        if self.num_spin > self.num_onmf_components:
            raise ValueError("num_spin must be less than or equal to num_onmf_components.")
        
        if not os.path.exists(self.save_path):
            raise ValueError("save_path does not exist.")

        self._epoch = epoch
        self._spin_thres = spin_thres
        self._stepsz = stepsz
        self._dropout = dropout
        self._counter = counter
        self._samplingsz = samplingsz
        self._samplingmix = samplingmix
        self._rec_gap = rec_gap
        self._lam_12h = lam_12h
        self._lam_l1j = lam_l1j
        
        self._matrix_std = None
        self._network = None
        self._responses = None
        self._onmf_rep_ori = None
        self._onmf_rep_tri = None
        self._sample_list = None
        self._onmf_summary = None
        self._raw_data = None
        self._gene_matrix_large = None
        self._use_data_list = None
        self.gene_program_csv = None


    @property
    def matrix_std(self):
        return self._matrix_std

    @property
    def network(self):
        return self._network

    @property
    def responses(self):
        return self._responses
    
    @property
    def onmf_rep_tri(self):
        return self._onmf_rep_tri
    
    @property
    def sample_list(self):
        return self._sample_list

    @property
    def onmf_summary(self):
        return self._onmf_summary
    
    @property
    def onmf_rep_ori(self):
        return self._onmf_rep_ori
    
    @matrix_std.setter
    def matrix_std(self, value):
        self._matrix_std = value
    
    @network.setter
    def network(self, value):
        self._network = value
    
    @responses.setter
    def responses(self, value):
        self._responses = value

    @onmf_rep_tri.setter
    def onmf_rep_tri(self, value):
        self._onmf_rep_tri = value

    @sample_list.setter
    def sample_list(self, value):
        self._sample_list = value

    @onmf_summary.setter
    def onmf_summary(self, value):
        self._onmf_summary = value

    @onmf_rep_ori.setter
    def onmf_rep_ori(self, value):
        self._onmf_rep_ori = value
    

    def matrix_balance(self):
        matrix_path_ori = prepare_onmf_decomposition(self.adata, self.save_path, balance_by='leiden', total_sample_size=2e4, method='squareroot')
        cur_matrix = np.load(matrix_path_ori)
        cur_std = cur_matrix.std(axis=0)
        cur_std = cur_std.clip(np.percentile(cur_std, 20), np.inf)
        cur_matrix /= cur_std
        self.gene_matrix_large = cur_matrix
        self._matrix_std = cur_std

    def onmf_abstract(self, balance_by='leiden', total_sample_size=2e4, method='squareroot'):
        adata = self.adata
        
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        
        maximum_sample_rate = 2
        cluster_list = list(adata.obs[balance_by].value_counts().keys())
        cluster_count = list(adata.obs[balance_by].value_counts())

        if method == 'porpotional':
            weight_fun = cluster_count
        elif method == 'squareroot':
            esti_size = (np.sqrt(cluster_count) / np.sum(np.sqrt(cluster_count)) * total_sample_size).astype(int)
            weight_fun = np.min([esti_size, maximum_sample_rate * np.array(cluster_count)], axis=0)
        elif method == 'equal':
            esti_size = total_sample_size / len(cluster_list)
            weight_fun = np.min([esti_size * np.ones(len(cluster_count)), maximum_sample_rate * np.array(cluster_count)], axis=0)
        sampling_number = (weight_fun / np.sum(weight_fun) * total_sample_size).astype(int)

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
        gene_matrix /= self._matrix_std
        onmf_rep_ori = self._onmf_summary.transform(gene_matrix)
        self.onmf_rep_ori = onmf_rep_ori

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
        # sample_corr_mean is the local one
        adata = self.adata
        onmf_rep_tri = self.onmf_rep_tri
        save_path = self.save_path
        raw_data, samp_list = sample_corr_mean(adata.obs[sample_col_name], onmf_rep_tri)
        self.raw_data = raw_data
        self.samp_list = samp_list
        
        filename = f"{save_path}data_raw.mat"
        savemat(filename, {'raw_data': raw_data, 'network_subset': list(range(len(samp_list))), 'samp_list': samp_list})

    
    def post_processing(self):
        # balance the experimental conditions by clustering and downsampling
        raw_data = self.raw_data
        fig_folder = self.save_path + 'figs/'
        use_data_list = select_diverse_sample(raw_data, num_cluster=32, fig_folder=fig_folder)
        self.use_data_list = use_data_list

    def network_construct(self, example_list=None):
        # parameter setting

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

        train_dat = {'cur_j': cur_j, 'cur_h': cur_h, 'epoch': self._epoch, 'spin_thres': self._spin_thres,
             'stepsz': self._stepsz, 'dropout': self._dropout, 'counter': self._counter,
             'samplingsz': self._samplingsz, 'samplingmix': self._samplingmix, 
             'rec_gap': self._rec_gap, 'task_name': task_name,
             'lam_12h': self._lam_12h, 'lam_l1j': self._lam_l1j}
        
        dir_list = [data_dir, data_dir + 'train_log']
        for directory in dir_list:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # compute the network        
        cur_j, cur_h = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat)

        self._network = cur_j
        self._responses = cur_h


    def network_infer(self, example_list=None):
        
        self.post_processing()
        self.network_construct(example_list=example_list)


    def state_list(self):
        adata = self.adata
        onmf_rep_tri = self.onmf_rep_tri
        samp_list = np.unique(adata.obs['sample_id'])

        state_list = []
        for samp in samp_list:
            cur_filt = adata.obs['sample_id'] == samp
            state_list.append(onmf_rep_tri[cur_filt, :] * 2 - 1) # why * 2 - 1?

    def pseudolikelihood(self):
        pass
        
    

    