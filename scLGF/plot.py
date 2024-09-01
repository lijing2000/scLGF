import pandas as pd
import argparse
from PlotClustering import plotClusteringImg4static
from utils import *
import scanpy as sc
import opt
import numpy as np
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='scDFCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # setting
    # Quake_10x_Limb_Muscle,Quake_Smart-seq2_Limb_Muscle,zeisel,Muraro,Romanov,Quake_Smart-seq2_Lung,Quake_Smart-seq2_Heart,Quake_10x_Bladder,Pollen,Young
    parser.add_argument('--name', type=str, default="Quake_Smart-seq2_Limb_Muscle")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--load_type', type=str, default='h5')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--highly_genes', type=int, default=2000)
    parser.add_argument('--k', type=int, default=15)
    args = parser.parse_args()


    file_path = "data/" + args.name + "." + args.load_type
    dataset = load_data_origin_data(file_path, args.load_type, scaling=True)
    x = dataset.x
    y = dataset.y
    adata = sc.AnnData(x)
    adata.obs['cell_labels'] = y
    #
    # x1 = dataset.x1
    # y1 = dataset.y1
    # adata_raw = sc.AnnData(x1)
    # adata_raw.obs['cell_labels'] = y1

    if args.name == 'Romanov':
        df1 = pd.read_csv('Romanov.csv')
        romanov = np.array(df1)
        y_dfcn_romanov = romanov[:,0]
        y_graphscc_romanov = romanov[:, 1]
        y_dec_romanov = romanov[:, 2]
        y_scdcc_romanov = romanov[:, 3]
        y_sctag_romanov = romanov[:, 4]
        y_scgae_romanov = romanov[:, 5]
        y_kmeans_romanov = romanov[:, 6]
        # print('DFCN:'+str(purity_score(y,y_dfcn_romanov)))
        # print('GraphSCC:' + str(purity_score(y, y_graphscc_romanov)))
        # print('DEC:' + str(purity_score(y, y_dec_romanov)))
        # print('scDCC:' + str(purity_score(y, y_scdcc_romanov)))
        # print('scTAG:' + str(purity_score(y, y_sctag_romanov)))
        # print('scGAE:' + str(purity_score(y, y_scgae_romanov)))
        # print('Kmeans:' + str(purity_score(y, y_kmeans_romanov)))

        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_romanov, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_romanov, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_romanov, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_romanov, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_romanov, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_romanov, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_romanov, 'umap')


    elif args.name == 'zeisel':
        df2 = pd.read_csv('zeisel.csv')
        zeisel = np.array(df2)
        y_dfcn_zeisel = zeisel[:, 0]
        y_graphscc_zeisel = zeisel[:, 1]
        y_dec_zeisel = zeisel[:, 2]
        y_scdcc_zeisel = zeisel[:, 3]
        y_sctag_zeisel = zeisel[:, 4]
        y_scgae_zeisel = zeisel[:, 5]
        y_kmeans_zeisel = zeisel[:, 6]
        # print('DFCN:' + str(purity_score(y, y_dfcn_zeisel)))
        # print('GraphSCC:' + str(purity_score(y, y_graphscc_zeisel)))
        # print('DEC:' + str(purity_score(y, y_dec_zeisel)))
        # print('scDCC:' + str(purity_score(y, y_scdcc_zeisel)))
        # print('scTAG:' + str(purity_score(y, y_sctag_zeisel)))
        # print('scGAE:' + str(purity_score(y, y_scgae_zeisel)))
        # print('Kmeans:' + str(purity_score(y, y_kmeans_zeisel)))

        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_zeisel, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_zeisel, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_zeisel, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_zeisel, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_zeisel, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_zeisel, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_zeisel, 'umap')






    elif args.name == 'Muraro':
        df3 = pd.read_csv('Muraro.csv')
        muraro = np.array(df3)
        y_dfcn_muraro = muraro[:, 0]
        y_graphscc_muraro = muraro[:, 1]
        y_dec_muraro = muraro[:, 2]
        y_scdcc_muraro = muraro[:, 3]
        y_sctag_muraro = muraro[:, 4]
        y_scgae_muraro = muraro[:, 5]
        y_kmeans_muraro = muraro[:, 6]

        # print('DFCN:' + str(purity_score(y, y_dfcn_muraro)))
        # print('GraphSCC:' + str(purity_score(y, y_graphscc_muraro)))
        # print('DEC:' + str(purity_score(y, y_dec_muraro)))
        # print('scDCC:' + str(purity_score(y, y_scdcc_muraro)))
        # print('scTAG:' + str(purity_score(y, y_sctag_muraro)))
        # print('scGAE:' + str(purity_score(y, y_scgae_muraro)))
        # print('Kmeans:' + str(purity_score(y, y_kmeans_muraro)))
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_muraro, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_muraro, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_muraro, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_muraro, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_muraro, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_muraro, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_muraro, 'umap')


    elif args.name == 'Quake_10x_Limb_Muscle':
        df4 = pd.read_csv('QX.csv')
        QX = np.array(df4)
        y_dfcn_QX = QX[:, 0]
        y_graphscc_QX = QX[:, 1]
        y_dec_QX = QX[:, 2]
        y_scdcc_QX = QX[:, 3]
        y_sctag_QX = QX[:, 4]
        y_scgae_QX = QX[:, 5]
        y_kmeans_QX = QX[:, 6]

        # print('DFCN:' + str(purity_score(y, y_dfcn_QX)))
        # print('GraphSCC:' + str(purity_score(y, y_graphscc_QX)))
        # print('DEC:' + str(purity_score(y, y_dec_QX)))
        # print('scDCC:' + str(purity_score(y, y_scdcc_QX)))
        # print('scTAG:' + str(purity_score(y, y_sctag_QX)))
        # print('scGAE:' + str(purity_score(y, y_scgae_QX)))
        # print('Kmeans:' + str(purity_score(y, y_kmeans_QX)))
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_QX, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_QX, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_QX, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_QX, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_QX, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_QX, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_QX, 'umap')


    elif args.name == 'Quake_Smart-seq2_Limb_Muscle':
        df5 = pd.read_csv('QS.csv')
        QS = np.array(df5)
        y_dfcn_QS = QS[:, 0]
        y_graphscc_QS = QS[:, 1]
        y_dec_QS = QS[:, 2]
        y_scdcc_QS = QS[:, 3]
        y_sctag_QS = QS[:, 4]
        y_scgae_QS = QS[:, 5]
        y_kmeans_QS = QS[:, 6]

        # print('DFCN:' + str(purity_score(y, y_dfcn_QS)))
        # print('GraphSCC:' + str(purity_score(y, y_graphscc_QS)))
        # print('DEC:' + str(purity_score(y, y_dec_QS)))
        # print('scDCC:' + str(purity_score(y, y_scdcc_QS)))
        # print('scTAG:' + str(purity_score(y, y_sctag_QS)))
        # print('scGAE:' + str(purity_score(y, y_scgae_QS)))
        # print('Kmeans:' + str(purity_score(y, y_kmeans_QS)))
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_QS, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_QS, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_QS, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_QS, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_QS, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_QS, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_QS, 'umap')

    elif args.name == 'Young':
        df5 = pd.read_csv('Young.csv')
        Young = np.array(df5)
        y_dfcn_Young = Young[:, 0]
        y_graphscc_Young = Young[:, 1]
        y_dec_Young = Young[:, 2]
        y_scdcc_Young = Young[:, 3]
        y_sctag_Young = Young[:, 4]
        y_scgae_Young = Young[:, 5]
        y_kmeans_Young = Young[:, 6]
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_Young, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_Young, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_Young, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_Young, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_Young, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_Young, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_Young, 'umap')

    elif args.name == 'Quake_Smart-seq2_Lung':
        df5 = pd.read_csv('QS_Lung.csv')
        Lung = np.array(df5)
        y_dfcn_Lung = Lung[:, 0]
        y_graphscc_Lung = Lung[:, 1]
        y_dec_Lung = Lung[:, 2]
        y_scdcc_Lung = Lung[:, 3]
        y_sctag_Lung = Lung[:, 4]
        y_scgae_Lung = Lung[:, 5]
        y_kmeans_Lung = Lung[:, 6]
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_Lung, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_Lung, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_Lung, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_Lung, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_Lung, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_Lung, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_Lung, 'umap')

    elif args.name == 'Quake_Smart-seq2_Heart':
        df5 = pd.read_csv('QS_Heart.csv')
        Heart = np.array(df5)
        y_dfcn_Heart =Heart[:, 0]
        y_graphscc_Heart = Heart[:, 1]
        y_dec_Heart = Heart[:, 2]
        y_scdcc_Heart = Heart[:, 3]
        y_sctag_Heart = Heart[:, 4]
        y_scgae_Heart = Heart[:, 5]
        y_kmeans_Heart = Heart[:, 6]
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_Heart, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_Heart, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_Heart, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_Heart, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_Heart, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_Heart, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_Heart, 'umap')


    elif args.name == 'Pollen':
        df5 = pd.read_csv('Pollen.csv')
        Pollen = np.array(df5)
        y_dfcn_Pollen =Pollen[:, 0]
        y_graphscc_Pollen = Pollen[:, 1]
        y_dec_Pollen = Pollen[:, 2]
        y_scdcc_Pollen = Pollen[:, 3]
        y_sctag_Pollen = Pollen[:, 4]
        y_scgae_Pollen = Pollen[:, 5]
        y_kmeans_Pollen = Pollen[:, 6]
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_Pollen, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_Pollen, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_Pollen, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_Pollen, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_Pollen, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_Pollen, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_Pollen, 'umap')

    elif args.name == 'Quake_10x_Bladder':
        df5 = pd.read_csv('Qx_Bladder.csv')
        Bladder = np.array(df5)
        y_dfcn_Bladder =Bladder[:, 0]
        y_graphscc_Bladder = Bladder[:, 1]
        y_dec_Bladder = Bladder[:, 2]
        y_scdcc_Bladder = Bladder[:, 3]
        y_sctag_Bladder = Bladder[:, 4]
        y_scgae_Bladder = Bladder[:, 5]
        y_kmeans_Bladder = Bladder[:, 6]
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        plotClusteringImg4static(adata, 'raw', y, 'umap')
        plotClusteringImg4static(adata, 'scDCSL', y_dfcn_Bladder, 'umap')
        plotClusteringImg4static(adata, 'GraphSCC', y_graphscc_Bladder, 'umap')
        plotClusteringImg4static(adata, 'scDeepCluster', y_dec_Bladder, 'umap')
        plotClusteringImg4static(adata, 'scDCC', y_scdcc_Bladder, 'umap')
        plotClusteringImg4static(adata, 'scTAG', y_sctag_Bladder, 'umap')
        plotClusteringImg4static(adata, 'scGAE', y_scgae_Bladder, 'umap')
        plotClusteringImg4static(adata, 'Kmeans', y_kmeans_Bladder, 'umap')


