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


from compute import compute_onmf, summarize_onmf_decomposition, corr_mean
from plotting import onmf_to_csv

def sample_corr_mean(samp_full, comp_bin):
    
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)
    
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])
        
    return raw_corr_data, samp_list

def onmf_abstract(adata: anndata.AnnData, 
                  save_path: str, 
                  num_spin: int = 10, 
                  num_pool: int = 10, 
                  num_repeat: int = 10) -> (np.ndarray, str):
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
    if num_spin > 10:
        warnings.warn("num_spin larger than 10 takes long time in Python. Please use computing clusters for larger num_spin.")

    if num_spin > num_pool:
        raise ValueError("num_spin must be less than or equal to num_pool.")
    
    if not os.path.exists(save_path):
        raise ValueError("save_path does not exist.")

    matrix = adata.X

    # Pre-computing num_repeat times
    print("Pre-computing")
    for seed in range(1, num_repeat + 1):
        print(f"Round_{seed}")
        current_onmf = compute_onmf(seed, num_spin, matrix)
        np.save(f"{save_path}onmf_{num_spin}_{seed}.npy", current_onmf)
    
    # Summarizing the ONMF result
    onmf_summary = summarize_onmf_decomposition(num_spin, num_repeat, num_pool, save_path, matrix)
    np.save(f"{save_path}onmf_summary_{num_spin}.npy", onmf_summary)

    # Save ONMF summary to CSV
    features = onmf_summary.components_
    gene_names = adata.var_names
    filename = onmf_to_csv(features, gene_names, save_path, thres=0.05)
    
    return onmf_summary, filename

def discretize(adata: anndata.AnnData) -> np.ndarray:
    """
    Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.
    
    Parameters:
    - adata (anndata.AnnData): The annotated data matrix.

    Returns:
    - np.ndarray: The discretized ONMF representation.
    """
    onmf_rep_ori = adata.X # may not be X
    if num_spin is None:
        num_spin = onmf_rep_ori.shape[1]
    num_gene = onmf_rep_ori.shape[1]

    sc.set_figure_params(figsize=[2, 2])
    fig, grid = sc.pl._tools._panel_grid(0.3, 0.3, ncols=7, num_panels=num_gene)
    onmf_rep_tri = np.zeros(onmf_rep_ori.shape)
    rec_kmeans = np.zeros(num_spin, dtype=object)

    
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
    
    return onmf_rep_tri

def cross_corr(adata: anndata.AnnData,
               onmf_rep_tri: np.ndarray,
               save_path: str) -> np.ndarray:
    # 'sample_id' is not robust enough
    # local sample_corr_mean
    raw_data, samp_list = sample_corr_mean(adata.obs['sample_id'], onmf_rep_tri)
    filename = f"{save_path}data_raw.mat"
    savemat(filename, {'raw_data': raw_data, 'network_subset': list(range(len(samp_list))), 'samp_list': samp_list})
    return raw_data


def network_infer(adata: anndata.AnnData,
                  raw_data: np.ndarray,
                  num_spin: int):
    # parameter setting

    num_samp = raw_data[0][0].shape[0]
    rec_all_corr = np.zeros((num_spin, num_spin, num_samp))
    rec_all_mean = np.zeros((num_spin, num_samp))

    for ii in range(num_samp):
        rec_all_corr[:, :, ii] = raw_data[ii][0]
        rec_all_mean[:, ii] = raw_data[ii][1].flatten()
    
    cur_j = np.zeros((num_spin, num_spin))
    cur_h = np.zeros((num_spin, num_samp))

    pass

def plot_jmat_network(G):

    self_loops = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(self_loops)

    eposi= [(u, v) for (u,v,d) in G.edges(data=True) if d['weight'] > 0]
    wposi= np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] > 0])

    enega = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < 0]
    wnega = np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] < 0])

    col1 = '#f0dab1'
    # nx.draw_networkx_nodes(G, pos, ax=ax, node_size=61.8 * nodesz, node_color=col1, edgecolors='None')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=61.8 * nodesz, node_color=node_color, edgecolors='k')

    sig_fun = lambda xx : (1 / (1 + np.exp(- 5 * (xx + cc))))
    cc = np.max(np.abs(j_mat)) / 10
    # edges
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=eposi, width=linewz * wposi,
                            edge_color='#3285CC', alpha=sig_fun(wposi))

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=enega, width=- linewz * wnega,
                            edge_color='#E84B23', alpha=sig_fun(- wnega))

    margin = 0.2
    plt.margins(x=0.1, y=0.1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    return ax
    