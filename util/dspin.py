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
    # summarize_onmf_decomposition, (the local one is used for test)
    corr_mean,
    learn_jmat_adam,
    prepare_onmf_decomposition,
    select_diverse_sample
)

from util.plotting import onmf_to_csv

def sample_corr_mean(samp_full, comp_bin):
    
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)
    
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])
        
    return raw_corr_data, samp_list

def summarize_onmf_decomposition(num_spin, num_repeat, num_pool, onmf_path, gene_matrices, fig_folder=None):
    
    # num_repeat = len(onmf_paths)
    # rec_components = np.zeros((num_repeat, num_pool, gene_matrix.shape[1]))

    # Assuming each matrix has the same shape as the first one in the list
    rec_components = np.zeros((num_repeat, num_pool, gene_matrices[0].shape[1]))
    
    for ii in range(num_repeat):
        cur_onmf = np.load('%sonmf_%d_%d.npy' % (onmf_path, num_pool, ii + 1), allow_pickle=True).item()
        rec_components[ii] = cur_onmf.components_
    
    all_components = rec_components.reshape(num_repeat * num_pool, -1)


    gene_weight = np.sum(all_components ** 2, axis=0) ** 0.5
    gene_sele_filt = gene_weight > np.mean(gene_weight)
    all_components_sub = all_components[:, gene_sele_filt]

    kmeans = KMeans(n_clusters=num_spin, random_state=0).fit(all_components)
    kmeans_gene = KMeans(n_clusters=num_spin, random_state=0).fit(all_components_sub.T)

    components_kmeans = np.zeros((num_spin, gene_matrices[0].shape[1]))
    for ii in range(num_spin):
        components_kmeans[ii] = np.mean(all_components[kmeans.labels_ == ii], axis=0)
    components_kmeans = normalize(components_kmeans, axis=1, norm='l2')
    
    components_summary = np.zeros((num_spin, gene_matrices[0].shape[1]))
    for ii in range(num_spin):
        filt_genes = np.argmax(components_kmeans, axis=0) == ii
        # Take the mean of the selected sub-matrix across all matrices
        sub_matrix = np.mean([m[:, filt_genes] for m in gene_matrices], axis=0)
        sub_onmf = NMF(n_components=1, init='random', random_state=0).fit(sub_matrix)
        components_summary[ii, filt_genes] = sub_onmf.components_[0]
    components_summary = normalize(components_summary, axis=1, norm='l2')
    
    onmf_summary = NMF(n_components=num_spin, init='random', random_state=0)
    onmf_summary.components_ = components_summary

    sc.set_figure_params(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    gene_order = np.argsort(kmeans_gene.labels_)
    comp_order = np.argsort(kmeans.labels_)
    plt.imshow(all_components_sub[comp_order, :][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(all_components) / 10, interpolation='none')
    plt.title('All components')

    plt.subplot(1, 3, 2)
    plt.imshow(components_kmeans[:, gene_sele_filt][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(components_kmeans) / 10, interpolation='none')
    plt.title('Kmeans components')

    plt.subplot(1, 3, 3)
    plt.imshow(components_summary[:, gene_sele_filt][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(all_components) / 10, interpolation='none')
    plt.title('Summary components')

    if fig_folder is not None:
        plt.savefig(fig_folder + 'onmf_decomposition_summary.png', dpi=300, bbox_inches='tight')

    return onmf_summary

class DSPIN:
    def __init__(self,
                 adata: anndata.AnnData,
                 save_path: str,
                 num_spin: int = 10,
                 num_pool: int = None,
                 num_repeat: int = 10,
                 epoch: int = 150,
                 spin_thres: int = 16,
                 stepsz: float = 0.02,
                 dropout: int = 0,
                 counter: int = 1,
                 samplingsz: float = 1e6,
                 samplingmix: float = 1e3,
                 rec_gap: int = 10,
                 lam_12h: float = 0.005,
                 lam_l1j: float = 0.01):
        self.adata = adata
        self.save_path = save_path
        
        self.num_spin = num_spin
        
        if num_pool is None:
            self.num_pool = num_spin
        else:
            self.num_pool = num_pool
        self.num_repeat = num_repeat

        num_gene = self.adata.X.shape[1]
        if num_gene > num_spin:
            self._onmf_indicator = True
        else:
            self._onmf_indicator = False
            self._onmf_rep_ori = adata.X

        if self.num_spin > 10:
            warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

        if self.num_spin > self.num_pool:
            raise ValueError("num_spin must be less than or equal to num_pool.")
        
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
    

    def preprocessing(self):
        matrix_path_ori = prepare_onmf_decomposition(self.adata, self.save_path, balance_by='leiden', total_sample_size=2e4, method='squareroot')
        cur_matrix = np.load(matrix_path_ori)
        cur_matrix /= cur_matrix.std(axis=0).clip(0.2, np.inf)
        self._matrix_std = cur_matrix.std(axis=0)
        return cur_matrix

    def onmf_abstract(self, balance_by='leiden', total_sample_size=2e4, method='squareroot') -> np.ndarray:
        """
        Abstracts the ONMF process: pre-computes ONMF multiple times, summarizes the results, and saves the summary to CSV.

        Parameters:
        - adata (anndata.AnnData): The annotated data matrix.
        - save_path (str): The path where the ONMF results and summary will be saved.
        - num_spin (int): Number of spins.
        - num_pool (int): Number of pools.
        - num_repeat (int), optional, default 10: Number of times to repeat the ONMF computation.

        Returns:
        - str: The filename where the ONMF summary has been saved as a CSV.
        """
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
        gene_matrixs = []

        # Pre-computing num_repeat times
        print("Pre-computing")
        for seed in range(1, self.num_repeat + 1):

            print(f"Round_{seed}")
            np.random.seed(seed) 
            for ii in tqdm(range(len(cluster_list))):
                cur_num = sampling_number[ii]
                cur_filt = adata.obs[balance_by] == cluster_list[ii]
                sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
                strart_ind = np.sum(sampling_number[:ii])
                end_ind = strart_ind + cur_num
                gene_matrix_balanced[strart_ind: end_ind, :] = adata.X[cur_filt, :][sele_ind, :]
                matrix_path = self.save_path + 'gmatrix_' + '{:.0e}'.format(total_sample_size) + '_balanced_' + method + '_' + str(seed) + '.npy'
                np.save(matrix_path, gene_matrix_balanced)

            gene_matrixs.append(gene_matrix_balanced)
            current_onmf = compute_onmf(seed, self.num_spin, gene_matrix_balanced)
            np.save(f"{self.save_path}onmf_{self.num_spin}_{seed}.npy", current_onmf)
        
        matrix = self.matrix

        # Summarizing the ONMF result
        onmf_summary = summarize_onmf_decomposition(self.num_spin, self.num_repeat, self.num_pool, 
                                                    onmf_path= self.save_path, 
                                                    gene_matrices= gene_matrixs,
                                                    fig_folder= self.save_path + 'fig/')
        np.save(f"{self.save_path}onmf_summary_{self.num_spin}.npy", onmf_summary)
        self.onmf_summary (onmf_summary)

        # Save ONMF summary to CSV
        features = onmf_summary.components_
        gene_names = self.adata.var_names
        filename = onmf_to_csv(features, gene_names, self.save_path, thres=0.05)

        return onmf_summary, filename
    
    def compute_onmf_rep_ori(self) -> np.ndarray:
        """
        Computes the original ONMF representation given the ONMF summary.
        """

        gene_matrix = self.adata.X.astype(np.float64)
        gene_matrix /= self._matrix_std
        onmf_rep_ori = self._onmf_summary.transform(gene_matrix)
        self._onmf_rep_ori = onmf_rep_ori

    def discretize(self) -> np.ndarray:
        """
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.

        Returns:
        - np.ndarray: The discretized ONMF representation.
        """
        onmf_rep_ori = self._onmf_rep_ori
        num_spin = onmf_rep_ori.shape[1]

        sc.set_figure_params(figsize=[2, 2])
        fig, grid = sc.pl._tools._panel_grid(0.3, 0.3, ncols=7, num_panels=num_spin)
        onmf_rep_tri = np.zeros(onmf_rep_ori.shape)
        rec_kmeans = np.zeros(self.num_spin, dtype=object)

        for ii in tqdm(range(num_spin)):
            ax = plt.subplot(grid[ii])
            km_fit = KMeans(n_clusters=3, n_init=10)
            km_fit.fit(onmf_rep_ori[:, ii].reshape(- 1, 1))
            plt.plot(np.sort(onmf_rep_ori[:, ii]));
            plt.plot(np.sort(km_fit.cluster_centers_[km_fit.labels_].reshape(- 1)));

            label_ord = np.argsort(km_fit.cluster_centers_.reshape(- 1))
            # the largest cluster is marked as 1, the smallest as -1, the middle as 0
            onmf_rep_tri[:, ii] = (km_fit.labels_ == label_ord[0]) * (-1) + (km_fit.labels_ == label_ord[2]) * 1
            rec_kmeans[ii] = km_fit
        
        self.onmf_rep_tri = onmf_rep_tri

        return onmf_rep_tri

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

        return raw_data
    
    def post_processing(self):
        # balance the experimental conditions by clustering and downsampling
        raw_data = self.raw_data
        fig_folder = self.save_path + 'figs/'
        use_data_list = select_diverse_sample(raw_data, num_cluster=32, fig_folder=fig_folder)
        self.use_data_list = use_data_list

    def network_infer(self, example_list=None):
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

                        
    

    