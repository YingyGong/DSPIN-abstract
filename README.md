# DSPIN

Tutorial, code and examples of the D-SPIN framework for the preprint "D-SPIN constructs gene regulatory network models from multiplexed scRNA-seq data revealing organizing principles of cellular perturbation response" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.19.537364))

![alternativetext](/figure/readme/Figure1_20230309_Inna.png)

## Installation

D-SPIN is implemented in Python and Matlab. The Python code is sufficient for analysis of small datasets with around 20k cells and tens of conditions. The Matlab code is only used for network inference on large datasets as Matlab can be easily deployed on clusters for parallelization with the build-in "parfor" function. 

The python code can be installed with [To be filled]

The Matlab code can be downloaded from the folder "DSPIN_matlab", and directly executable in Matlab after specifying the path to the data. 

## Dependencies

DSPIN requires the following python packages:

[To be filled]

## D-SPIN Overview

D-SPIN contains three main steps: 
* Gene program discovery
* Network inference
* Network analysis

The input data should be AnnData object after typical single-cell preprocessing steps including cell counts normalization, log transformation, highly-varible gene selection, and clustering (optional). The input data should contain the following information. The attribute name is subject to the user while the following are defaults of the code:
* Normalized, log-transformed and filtered gene expression matrix in adata.X
* Sample information for each cell in adata.obs['sample_id']
* Batch information for each cell in adata.obs['batch']
* Clustering information for each cell in adata.obs['leiden']
* If some samples are control conditions, the relative response vectors will be computed for each batch with respect to the average of control conditions. The control conditions can be indicated in adata.obs['if_control'] with 1 for control and 0 for non-control.

### Gene program discovery

By default, D-SPIN use orgthogonal non-negative matrix factorization (ONMF) to discover gene programs that coexpress in the data. The user can also specify pre-defined gene programs in full or partial. Specifically, 

In the gene program discovery step, D-SPIN takes the following arguments: 
[To be filled]. 

### Network inference

In the network inference step, D-SPIN automatically choose between three inference methods depending on the number of gene programs: (1) Exact maximum-likelihood inference (2) Markov-Chain Monte-Carlo (MCMC) maximum-likelihood inference (3) Pseudo-likelihood inference. The inference method can also be specified by the user. D-SPIN takes the following arguments [To be filled]

In the network analysis step, D-SPIN identifies the modules in the inferred network, and automatically provide a layout of the network and perturbation responses. Further analysis and interpretaion of the network and perturbation response is up to the custome purpose. 

## Application to signaling response data of human PBMCs


## D-SPIN Demo

D-SPIN takes single-cell sequencing data of multiple perturbation conditions. In the second demo, PBMCs are treated with different signaling molecules such as CD3 antibody, LPS, IL1B, and TGFB1

![alternativetext](/figure/thomsonlab_signaling/example_conditions.png)

D-SPIN identifies a set of gene programs that coexpress in the data, and represent each cell as a combination of gene program expression states. 

![alternativetext](/figure/thomsonlab_signaling/gene_program_example.png)

D-SPIN uses cross-correlation and mean of each perturbation condition to inferred a unified regulatory network and the response vector of each perturbation condition. The inference can be parallelized across perturbation conditions. The inference code is in Matlab using "parfor", while for demo purpose Python code (without parallelization) is provided.

The inferred regulatory network and perturbations can be jointly analyzed to reveal how perturbations act in the context of the regulatory network.

![alternativetext](/figure/thomsonlab_signaling/joint_network_perturbation.png)

## Demos

Two demos of D-SPIN are available on Google Colab. 

The first demo reconstructs the regulatory network of simulated hematopoietic stem cell (HSC) differentiation network with perturbations using the BEELINE framework (Pratapa, Aditya, et al. Nature methods, 2020). 

[Demo1](https://colab.research.google.com/drive/1YdvjNiCkyGx-azXzXz7gqjGGE9RXrDbL?usp=sharing)

The second demo reconstructs regulatory network and response vector in a single-cell dataset collected in the ThomsonLab.In the dataset, human peripheral blood mononuclear cells (PBMCs) were treated with various signaling molecules with different dosages. 

[Demo2](https://colab.research.google.com/drive/1zrWFZWtaHQAzG88jgtovCPzt3wiXdlwf?usp=sharing)

# General suggestions for using D-SPIN

# References

1. Pratapa, Aditya, et al. "Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data." Nature methods 17.2 (2020): 147-154.
