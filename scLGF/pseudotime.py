import scanpy as sc
from loadData import loadAnnData
import matplotlib.pyplot as pt
import opt
from utils import *
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt




def pseudotime(adata_raw,adata,cluster_results):
    # adata.obs[title] = pd.Categorical(
    #     values=cluster_results.astype('U'),
    #     categories=natsorted(map(str, np.unique(cluster_results))),
    # )

    ## sc原始数据绘图

    # sc.pp.neighbors(adata, n_pcs=10)
    # # sc.pl.scatter(adata, basis="tsne", color="clusters")
    # sc.tl.diffmap(adata)
    # root_ixs = adata.obsm["X_diffmap"][:, 3].argmin()
    # sc.pl.scatter(
    #     adata,
    #     basis="tsne",
    #     color=["dpt_pseudotime", "palantir_pseudotime"],
    #     color_map="gnuplot2",
    # )

    sc.tl.louvain(adata)  # 可以使用resolution调节聚类的簇的数据，如resolution=1.0
    sc.tl.paga(adata, groups='louvain')
    sc.pl.paga(adata, color=['louvain', 'MS4A1', 'NKG7', 'PPBP'])  ##随便挑选了几个基因
    pt.savefig("paga_celltype.pdf")

    # adata.uns["iroot"] = root_ixs
    # sc.tl.dpt(adata)



if __name__ == '__main__':
    file_path = "data/" + opt.args.name + "." + opt.args.load_type
    dataset = load_data_origin_data(file_path, opt.args.load_type, scaling=True)
    x = dataset.x
    y = dataset.y
    adata_raw = sc.AnnData(x)
    adata_raw.obs['cell_labels'] = y

    pseudotime(adata_raw,adata_raw,y)

