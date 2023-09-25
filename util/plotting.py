# -*-coding:utf-8 -*-
'''
@Time    :   2023/04/06 15:00
@Author  :   Jialong Jiang
'''

import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import leidenalg as la
import igraph as ig
from pyvis import network as net

import csv 

def onmf_to_csv(features, gene_name, data_folder, thres=0.01):

    num_spin = features.shape[0]
    rec_max_gene = 0
    with open(data_folder + 'onmf_gene_list_%d.csv' % num_spin,'w', newline='') as file:
        write = csv.writer(file)
        for spin in range(num_spin):
            num_gene_show = np.sum(features[spin, :] > thres)
            rec_max_gene = max(rec_max_gene, num_gene_show)
            gene_ind = np.argsort(- features[spin, :])[: num_gene_show]
            cur_line = list(gene_name[gene_ind])
            cur_line.insert(0, spin)
            write.writerow(cur_line) 

    pd_csv = pd.read_csv(data_folder + 'onmf_gene_list_%d.csv' % num_spin, names=range(rec_max_gene + 1))
    pd_csv = pd_csv.transpose()
    pd_csv.to_csv(data_folder + 'onmf_gene_list_%d.csv' % num_spin, header=False, index=False)
    return(data_folder + 'onmf_gene_list_%d.csv' % num_spin)

def onmf_gene_program_info(features, gene_name, num_gene_show, fig_folder=None):

    thres = 0.01
    num_spin = features.shape[0]
    sc.set_figure_params(figsize=[1.5, 3.6])
    fig, grid = sc.pl._tools._panel_grid(0.26, 0.9, ncols=6, num_panels=num_spin)

    for spin in range(num_spin):
        ax = plt.subplot(grid[spin])

        cur_num_gene_show = min(num_gene_show, np.sum(features[spin, :] > thres))
        gene_ind = np.argsort(- features[spin, :])[: cur_num_gene_show]
        plt.plot(features[spin, gene_ind], np.arange(cur_num_gene_show), 'o')
        plt.grid()
        plt.xlim([0, 1.1 * np.max(features[spin, gene_ind])])
        plt.ylim([-0.5, num_gene_show - 0.5])
        plt.yticks(np.arange(cur_num_gene_show), gene_name[gene_ind], fontsize=9);
        plt.gca().invert_yaxis()

        plt.title(spin)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'onmf_gene_program_info.png', dpi=300, bbox_inches='tight')

    return(fig_folder + 'onmf_gene_program_info.png')

from scipy import optimize
from scipy.spatial.distance import pdist
def assign_program_position(onmf_rep_ori, umap_all, repulsion=2):

    num_spin = onmf_rep_ori.shape[1]
    program_umap_pos = np.zeros([num_spin, 2])
    for ii in range(num_spin):
        weight_sub = onmf_rep_ori[:, ii]
        weight_sub[weight_sub < np.percentile(weight_sub, 99.5)] = 0
        program_umap_pos[ii, :] = np.sum(umap_all * weight_sub.reshape(- 1, 1) / np.sum(weight_sub), axis=0)

    ori_pos = program_umap_pos.copy()
    def layout_loss_fun(xx):
        xx = xx.reshape(- 1, 2)
        attract = np.sum((xx - ori_pos) ** 2)
        repulse = np.sum(repulsion / pdist(xx))
        return attract + repulse

    opt_res = optimize.minimize(layout_loss_fun, program_umap_pos.flatten())
    program_umap_pos = opt_res.x.reshape(- 1, 2)

    sc.set_figure_params(figsize=[4, 4])

    
    if umap_all.shape[0] <= 5e4:
        plt.scatter(umap_all[:, 0], umap_all[:, 1], s=1, c='#bbbbbb', alpha=min(1, 1e4 / umap_all.shape[0]))
    else:
        sele_ind = np.random.choice(umap_all.shape[0], size=50000, replace=False).astype(int)
        plt.scatter(umap_all[sele_ind, 0], umap_all[sele_ind, 1], s=1, c='#bbbbbb', alpha=0.2)

    for ii in range(num_spin):
        plt.text(program_umap_pos[ii, 0], program_umap_pos[ii, 1], str(ii), fontsize=10)
    plt.axis('off')

    return program_umap_pos

import matplotlib.patheffects as PathEffects
def gene_program_on_umap(onmf_rep, umap_all, program_umap_pos, fig_folder=None, subsample=True):

    num_spin = onmf_rep.shape[1]

    if subsample:
        num_subsample = 20000
        sub_ind = np.random.choice(onmf_rep.shape[0], num_subsample, replace=False)
        onmf_rep = onmf_rep[sub_ind, :]
        umap_all = umap_all[sub_ind, :]

    sc.set_figure_params(figsize=[2, 2])
    fig, grid = sc.pl._tools._panel_grid(0.2, 0.06, ncols=6, num_panels=num_spin)
    for spin in range(num_spin):
        ax = plt.subplot(grid[spin])

        plt.scatter(umap_all[:, 0], umap_all[:, 1], c=onmf_rep[:, spin], s=1, 
        alpha=0.5, vmax=1, cmap='BuPu', vmin=-0.1)
        plt.text(program_umap_pos[spin, 0], program_umap_pos[spin, 1], str(spin), fontsize=12, path_effects=[PathEffects.withStroke(linewidth=3, foreground='w')])
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(spin)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'gene_program_on_umap.png', dpi=300, bbox_inches='tight')

    return(fig_folder + 'gene_program_on_umap.png')

# this one is the network with irregular shape
def plot_j_network(j_mat, 
                   pos=None, 
                   label=None, 
                   thres=None, 
                   seed=0,
                   node_size: float = 0.2,
                   label_dist: float = 0.01,
                   line_width: float = 10):
    
    if thres is not None:
        j_filt = j_mat.copy()
        j_filt[np.abs(j_mat) < thres] = 0
        np.fill_diagonal(j_filt, 0)
        j_mat = j_filt
        
    ax = plt.gca()

    G = nx.from_numpy_array(j_mat)

    eposi= [(u, v) for (u,v,d) in G.edges(data=True) if d['weight'] > 0]
    wposi= np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] > 0])

    enega = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < 0]
    wnega = np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] < 0])
    
    if pos is None:
        pos = nx.spring_layout(G, weight=1, seed=seed)
        # pos = nx.spectral_layout(G)

    col1 = '#f0dab1'
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=61.8 * node_size, node_color=col1, edgecolors=None)
    if label is not None:
        nx.draw_networkx_labels(G, pos, labels=label, font_size=20)
    
    sig_fun = lambda xx : (1 / (1 + np.exp(- 5 * (xx + cc))))
    cc = - np.max(np.abs(j_mat)) / 4
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=eposi, width=line_width * wposi, 
                           edge_color='#9999ff', alpha=sig_fun(wposi))

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=enega, width=-line_width * wnega, 
                           edge_color='#ff9999', alpha=sig_fun(- wnega))
    ax.set_axis_off()
        
    return 



def temporary_spin_name(csv_file, num_gene: int = 4): 
    df = pd.read_csv(csv_file, header=None)
    spin_names = ['P' + '_'.join(map(str, df[col][:6])) for col in df.columns]
    return spin_names

def format_label(label):
    parts = label.split('_')
    i = 0
    while i < len(parts) - 1:
        if i % 2 == 1:
            parts[i] = parts[i] + '\n'
        i += 1

    return '_'.join(parts)


import matplotlib.patheffects as patheffects

def plot_final(gene_program_name, cur_j,
               nodesz: float = 3, linewz: float = 1, node_color: str = 'k', pos=None):
    
    def spin_order_in_cluster(j_mat):
        np.fill_diagonal(j_mat, 0)
        
        thres = 0
        j_filt = j_mat.copy()
        j_filt[np.abs(j_mat) < thres] = 0
        np.fill_diagonal(j_filt, 0)
        G = nx.from_numpy_array(j_filt)

        G = ig.Graph.from_networkx(G)
        G_pos = G.subgraph_edges(G.es.select(weight_gt = 0), delete_vertices=False);
        G_neg = G.subgraph_edges(G.es.select(weight_lt = 0), delete_vertices=False);
        G_neg.es['weight'] = [-w for w in G_neg.es['weight']]

        part_pos = la.RBConfigurationVertexPartition(G_pos, weights='weight', resolution_parameter=2)
        part_neg = la.RBConfigurationVertexPartition(G_neg, weights='weight', resolution_parameter=2)
        optimiser = la.Optimiser()
        diff = optimiser.optimise_partition_multiplex([part_pos, part_neg],layer_weights=[1,-1]);

        net_class = list(part_pos)
        spin_order = [spin for cur_list in net_class for spin in cur_list]
        net_class_len = [len(cur_list) for cur_list in net_class]

        start_angle = 0 * np.pi
        end_angle = 2 * np.pi
        gap_size = 2


        angle_list_raw = np.linspace(start_angle, end_angle, np.sum(net_class_len) + gap_size * len(net_class_len) + 1)[: - 1]
        angle_list = []
        size_group_cum = np.cumsum(net_class_len)
        size_group_cum = np.insert(size_group_cum, 0, 0)
        # angle_list = np.linspace(start_angle, end_angle, len(leiden_list) + 1)
        for ii in range(len(net_class_len)):
            angle_list.extend(angle_list_raw[size_group_cum[ii] + gap_size * ii: size_group_cum[ii + 1] + gap_size * ii])
            
        pert_dist = 3

        pert_pos = np.array([- pert_dist * np.cos(angle_list), pert_dist * np.sin(angle_list)]).T

        return spin_order, pert_pos
    
    def plot_network(G, j_mat, ax, nodesz=1, linewz=1, node_color='k', pos=None): 
    
        self_loops = [(u, v) for u, v in G.edges() if u == v]
        G.remove_edges_from(self_loops)

        eposi= [(u, v) for (u,v,d) in G.edges(data=True) if d['weight'] > 0]
        wposi= np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] > 0])

        enega = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < 0]
        wnega = np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] < 0])

        col1 = '#f0dab1'
        
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
    
    def adjust_label_position(pos, offset=0.1):
        """Move labels radially outward from the center by a given offset."""
        adjusted_pos = {}
        for node, coordinates in enumerate(pos):
            theta = np.arctan2(coordinates[1], coordinates[0])
            radius = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
            adjusted_pos[node] = (coordinates[0] + np.cos(theta)*offset, coordinates[1] + np.sin(theta)*offset)
        return adjusted_pos

    sc.set_figure_params(figsize=[40, 40])

    spin_order, pert_pos = spin_order_in_cluster(cur_j)

    num_spin = cur_j.shape[0]

    fig, grid = sc.pl._tools._panel_grid(0.2, 0.2, ncols=2, num_panels=2)

    cur_j_filt = cur_j.copy()
    cur_j_filt[np.abs(cur_j_filt) < np.percentile(np.abs(cur_j_filt), 40)] = 0

    G = nx.from_numpy_matrix(cur_j_filt[spin_order, :][:, spin_order])

    pos = pert_pos
    node_color = ['#f0dab1'] * num_spin
    node_label = np.array(gene_program_name)[spin_order]
    # node_label = np.array([format_label(label) for label in gene_list])

    nodesz = np.sqrt(100 / num_spin)
    linewz = np.sqrt(100 / num_spin)

    ax = plt.subplot(grid[1])
    ax = plot_network(G, cur_j_filt, ax, nodesz=nodesz, linewz=linewz, node_color=node_color, pos=pos)

    path_effect = [patheffects.withStroke(linewidth=3, foreground='w')]

    adjusted_positions = adjust_label_position(pos, 0.5)
    for ii in range(num_spin):
        x, y = adjusted_positions[ii]
        text = plt.text(x, y, node_label[ii], fontsize=1000/num_spin, color='k', ha='center', va='center', rotation=np.arctan(pos[ii][1] / pos[ii][0]) / np.pi * 180)
        text.set_path_effects(path_effect)
    ax.set_title('Gene Regulatory Network under Cancerous Conditions')

from sklearn.cluster import KMeans

def gene_program_decomposition(onmf_summary,
                               num_spin,
                               spin_name_extend,
                               gene_matrix,
                               onmf_rep_tri,
                               fig_folder,
                               num_gene_select: int = 10,
                               n_clusters: int = 4):
    features = onmf_summary.components_
    num_gene_select = num_gene_select
    gene_mod_ind = np.argmax(features, axis=0)

    gene_mod_use = []
    for ind in range(num_spin):
        ii = ind
        gene_in_mod = np.where(gene_mod_ind == ii)[0]
        cur_gene = gene_in_mod[np.argsort(- features[ii, gene_in_mod])[: num_gene_select]]
        gene_mod_use += list(cur_gene)
    gene_mod_use = np.array(gene_mod_use)

    np.random.seed(0)
    subset_ind = np.random.choice(range(gene_matrix.shape[0]), size=10000, replace=False)
    cell_order = np.argsort(KMeans(n_clusters=n_clusters).fit_predict(onmf_rep_tri[subset_ind, :]))
    gene_matrix_subset = gene_matrix[subset_ind, :][:, gene_mod_use]
    gene_matrix_subset /= np.max(gene_matrix, axis=0)[gene_mod_use].clip(0.2, np.inf)

    sc.set_figure_params(figsize=[10, 5])

    plt.subplot(1, 2, 1)
    plt.imshow(gene_matrix_subset[cell_order, :].T, aspect='auto', cmap='Blues', interpolation='none')
    plt.ylabel('Gene')
    plt.xlabel('Cell')
    plt.title('Gene expression')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.imshow(onmf_rep_tri[subset_ind, :][cell_order, :].T, aspect='auto', cmap='Blues', interpolation='none')
    plt.yticks(range(num_spin), spin_name_extend, fontsize=12)
    plt.gca().yaxis.set_ticks_position('right')
    plt.xlabel('Cell');
    plt.title('Gene program expression')
    plt.grid()

    plt.savefig(fig_folder + 'gene_program_decomposition.png', bbox_inches='tight')

def node_cluster():
    start_angle = 0 * np.pi
    end_angle = 2 * np.pi
    gap_size = 2


    angle_list_raw = np.linspace(start_angle, end_angle, np.sum(net_class_len) + gap_size * len(net_class_len) + 1)[: - 1]
    angle_list = []
    size_group_cum = np.cumsum(net_class_len)
    size_group_cum = np.insert(size_group_cum, 0, 0)
    # angle_list = np.linspace(start_angle, end_angle, len(leiden_list) + 1)
    for ii in range(len(net_class_len)):
        angle_list.extend(angle_list_raw[size_group_cum[ii] + gap_size * ii: size_group_cum[ii + 1] + gap_size * ii])
        
    pert_dist = 3

    pert_pos = np.array([- pert_dist * np.cos(angle_list), pert_dist * np.sin(angle_list)]).T
