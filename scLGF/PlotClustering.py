import scanpy as sc
from loadData import loadAnnData


import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt




def plotClusteringImg4static(adata,title,cluster_results,type='umap'):
    adata.obs[title] = pd.Categorical(
        values=cluster_results.astype('U'),
        categories=natsorted(map(str, np.unique(cluster_results))),
    )

    ## sc原始数据绘图
    # 如果设置了 adata 的 .raw 属性时，下图显示了“raw”（标准化、对数化但未校正）基因表达矩阵。
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    #sc.tl.umap(adata)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4), gridspec_kw={'wspace': 0.9})
    sc.pl.umap(adata, color=title, ax=ax1, show=False)
    plt.savefig(title+".png",bbox_inches='tight') # 注意两个参数
    # plt.savefig(title + ".png")
    plt.show()


def plotClusteringImg(adata,title,cluster_results,type='umap'):
    adata.obs[title] = pd.Categorical(
        values=cluster_results.astype('U'),
        categories=natsorted(map(str, np.unique(cluster_results))),
    )

    ## sc原始数据绘图
    # 如果设置了 adata 的 .raw 属性时，下图显示了“raw”（标准化、对数化但未校正）基因表达矩阵。
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)



    if type=='tsne':
        sc.tl.tsne(adata)
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4), gridspec_kw={'wspace': 0.9})
        sc.pl.tsne(adata, color=title, ax=ax1, show=False)
        plt.savefig(title + ".pdf", bbox_inches='tight')  # 注意两个参数


        #plt.savefig(title + ".png")
        plt.show()
    elif type=='umap':
        sc.tl.umap(adata)

        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4), gridspec_kw={'wspace': 0.9})
        sc.pl.umap(adata, color=title, ax=ax1, show=False)
        plt.savefig(title+".pdf",bbox_inches='tight') # 注意两个参数
        # plt.savefig(title + ".png")
        plt.show()

    # sc.tl.leiden(adata)
    # adata.obs['cell_labels'] = cluster_results
    # marker_genes = ['14465', '23770', '7905', '23625', '30179', '18011', '30846',
    #                 '9822', '948', '4005', '27694', '10688',
    #                 '25624', '5089', '30065', '15635', '30197','1906']
    # sc.pl.dotplot(adata, marker_genes, groupby= 'cell_labels' );
    # # sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    # # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

#
# if __name__ == '__main__':
#
#     dataset = "biase"
#
#     dataPath = "./data/" + dataset + ".csv"
#
#     dataset = dataset + "_run"
#     adata = loadAnnData(dataPath)
#
#     cluster_results = adata.obs['cluster_labels']
#
#     plotClusteringImg(adata,adata.X,"SNNDCP",cluster_results)





