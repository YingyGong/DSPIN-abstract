import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
import anndata
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.io import savemat, loadmat
import networkx as nx
import matplotlib.patheffects as patheffects
import warnings


from util.compute import compute_onmf, summarize_onmf_decomposition, corr_mean, learn_jmat_adam
from util.plotting import onmf_to_csv

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
                 num_pool: int = 10,
                 num_repeat: int = 10):
        self.adata = adata
        self.save_path = save_path
        self.num_spin = num_spin
        self.num_pool = num_pool
        self.num_repeat = num_repeat

        if self.num_spin > 10:
            warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

        if self.num_spin > self.num_pool:
            raise ValueError("num_spin must be less than or equal to num_pool.")
        
        if not os.path.exists(self.save_path):
            raise ValueError("save_path does not exist.")


    @property
    def network(self):
        return self._network

    @property
    def responses(self):
        return self._responses
    
    @network.setter
    def network(self, value):
        self._network = value
    
    @responses.setter
    def responses(self, value):
        self._responses = value

    def onmf_abstract(self) -> np.ndarray:
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
        
        matrix = self.adata.X

        # Pre-computing num_repeat times
        print("Pre-computing")
        for seed in range(1, self.num_repeat + 1):
            print(f"Round_{seed}")
            current_onmf = compute_onmf(seed, self.num_spin, matrix)
            np.save(f"{self.save_path}onmf_{self.num_spin}_{seed}.npy", current_onmf)
        
        # Summarizing the ONMF result
        onmf_summary = summarize_onmf_decomposition(self.num_spin, self.num_repeat, self.num_pool, self.save_path, matrix)
        np.save(f"{self.save_path}onmf_summary_{self.num_spin}.npy", onmf_summary)

        # Save ONMF summary to CSV
        features = onmf_summary.components_
        gene_names = self.adata.var_names
        filename = onmf_to_csv(features, gene_names, self.save_path, thres=0.05)

        self.onmf_summary = onmf_summary
        
        return onmf_summary, filename
    
    def set_onmf_summary(self, onmf_summary: np.ndarray):
        self.onmf_summary = onmf_summary

    def discretize(self) -> np.ndarray:
        """
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.
        
        Parameters:
        - adata (anndata.AnnData): The annotated data matrix.

        Returns:
        - np.ndarray: The discretized ONMF representation.
        """
        onmf_rep_ori = self.onmf_summary
        num_gene = onmf_rep_ori.shape[1]

        sc.set_figure_params(figsize=[2, 2])
        fig, grid = sc.pl._tools._panel_grid(0.3, 0.3, ncols=7, num_panels=num_gene)
        onmf_rep_tri = np.zeros(onmf_rep_ori.shape)
        rec_kmeans = np.zeros(self.num_spin, dtype=object)

        for ii in tqdm(range(num_gene)):
            ax = plt.subplot(grid[ii])
            km_fit = KMeans(n_clusters=3, n_init=10)
            km_fit.fit(onmf_rep_ori[:, ii].reshape(- 1, 1))
            plt.plot(np.sort(onmf_rep_ori[:, ii]));
            plt.plot(np.sort(km_fit.cluster_centers_[km_fit.labels_].reshape(- 1)));

            label_ord = np.argsort(km_fit.cluster_centers_.reshape(- 1))
            # the largest cluster is marked as 1, the smallest as -1, the middle as 0
            onmf_rep_tri[:, ii] = (km_fit.labels_ == label_ord[1]) * (-1) + (km_fit.labels_ == label_ord[2]) * 1
            rec_kmeans[ii] = km_fit
        
        self.onmf_rep_tri = onmf_rep_tri

        return onmf_rep_tri


    def cross_corr(self) -> np.ndarray:
        # unsure whether 'sample_id' is robust enough
        # sample_corr_mean has a small difference
        adata = self.adata
        onmf_rep_tri = self.onmf_rep_tri
        save_path = self.save_path
        raw_data, samp_list = sample_corr_mean(adata.obs['sample_id'], onmf_rep_tri)
        filename = f"{save_path}data_raw.mat"
        savemat(filename, {'raw_data': raw_data, 'network_subset': list(range(len(samp_list))), 'samp_list': samp_list})

        self.raw_data = raw_data
        return raw_data
    

    def network_infer(self):
        # parameter setting

        raw_data = self.raw_data
        num_spin = self.num_spin

        num_samp = raw_data[0][0].shape[0]
        rec_all_corr = np.zeros((num_spin, num_spin, num_samp))
        rec_all_mean = np.zeros((num_spin, num_samp))

        for ii in range(num_samp):
            rec_all_corr[:, :, ii] = raw_data[ii][0]
            rec_all_mean[:, ii] = raw_data[ii][1].flatten()
        
        cur_j = np.zeros((num_spin, num_spin))
        cur_h = np.zeros((num_spin, num_samp))
        data_dir = self.save_path + 'dspin_python/'
        task_name = data_dir + 'train_log'
        train_dat = {'cur_j': cur_j, 'cur_h': cur_h, 'epoch': 200, 'spin_thres': 16,
             'stepsz': 0.2, 'dropout': 0, 'counter': 1,
             'samplingsz': 5e7, 'samplingmix': 1e3, 'rec_gap': 10, 'task_name': task_name}
        
        dir_list = [data_dir, data_dir + 'train_log']
        for directory in dir_list:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        cur_j, cur_h = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat)

        self._network = cur_j
        self._responses = cur_h

                        
    

    