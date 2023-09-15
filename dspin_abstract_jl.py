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

class AbstractDSPIN(ABC):
    def __init__(self, 
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        
        self._onmf_rep_ori = None
        self._onmf_rep_tri = None
        self._raw_data = None
        self._network = None
        self._responses = None

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
            super().__init__(adata, save_path, num_spin, num_onmf_components, num_repeat)

            print("LargeDSPIN initialized.")

            self._onmf_summary = None
            self._gene_matrix_large = None
            self._use_data_list = None
            self.gene_program_csv = None
            self.preprograms = None
            self.preprogram_num = len(preprograms) if preprograms else 0
                
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
    
    data_folder = 'data/HSC_simulation/'

    cadata = ad.read_h5ad(data_folder + 'hsc_simulation_with_perturbations.h5ad')

    random_select = np.random.choice(cadata.shape[0], 10000, replace=False)
    cadata = cadata[random_select, :]