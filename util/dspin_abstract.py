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
    prepare_onmf_decomposition,
    select_diverse_sample,
    onmf_discretize,
    sample_corr_mean
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
                 num_repeat: int = 10):
        self.adata = adata
        self.save_path = save_path
        self.num_spin = num_spin
        
        if num_onmf_components is None:
            self.num_onmf_components = num_spin
        else:
            self.num_onmf_components = num_onmf_components
        self.num_repeat = num_repeat

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
    pass


# Select the class to use
class DSPIN(object):
    def __new__(cls, *args, **kwargs):
        if args[0].shape[0] < 100000:
            return SmallDSPIN(*args, **kwargs)
        else:
            return LargeDSPIN(*args, **kwargs)