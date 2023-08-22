import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
import anndata
import os
from sklearn.cluster import KMeans
import tqdm
from scipy.io import savemat, loadmat
import networkx as nx
import matplotlib.patheffects as patheffects
import warnings


from compute import compute_onmf, summarize_onmf_decomposition, sample_corr_mean
from plotting import onmf_to_csv

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
        if self.num_spin > 10:
            warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

        if self.num_spin > self.num_pool:
            raise ValueError("num_spin must be less than or equal to num_pool.")
        
        if not os.path.exists(self.save_path):
            raise ValueError("save_path does not exist.")

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
        
        return onmf_summary, filename
        
                      

    