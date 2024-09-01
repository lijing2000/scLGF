import numpy as np
import pandas as pd
import scanpy as sc
from preprocess import normalize
import opt
# from __future__ import print_function, division
from utils import *
import opt
from  scDFCN import scDFCN
from train import train, acc_reuslt, nmi_result, f1_result, ari_result
from PlotClustering import plotClusteringImg,plotClusteringImg4static
from preprocess import normalize
import time


if __name__ == '__main__':

    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    adata = sc.read_10x_mtx(
        'data/hg19/',  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)                              # wr
    adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
    # sc.pl.highest_expr_genes(adata, n_top=20, )
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    sc.pp.normalize_total(adata, target_sum=1e4)

    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)

    adata = adata[:, adata.var.highly_variable]

    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver='arpack')


    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')

    sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)

    sc.tl.leiden(adata)
    sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])

    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

    # adata_nom = normalize(adata, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    # x = adata_nom.X
    # y = adata_nom.obs['obs_names']
    # A, A_norm = load_graph(x)
    #
    # x = numpy_to_torch(x).to(opt.args.device)
    # A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    # model = scDFCN(n_node=x.shape[0]).to(opt.args.device)
    # print(model)
    #
    #
    # pretrain(model,LoadDataset(x), A_norm)
    # adata_embedding,cluster_lables, acc_reuslt,nmi_result,ari_result = train(model, LoadDataset(x), x, y, A, A_norm)


