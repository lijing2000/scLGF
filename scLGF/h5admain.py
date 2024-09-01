from __future__ import print_function, division
from utils import *
import opt
import scanpy
from  scDFCN import scDFCN
from train import train, acc_reuslt, nmi_result, f1_result, ari_result
from PlotClustering import plotClusteringImg,plotClusteringImg4static
from preprocess import normalize
if __name__ == "__main__":
    #
    print(opt.args.name)
    setup()

    # data_mat = h5py.File(opt.args.data_file)
    # x = np.array(data_mat['X'])
    # y = np.array(data_mat['Y'])
    # data_mat.close()
    adata = scanpy.read_h5ad("simulation/sim1.h5ad")
    x = adata.raw.X
    group = pd.read_csv("simulation/sim1.csv")
    y = np.array(group)
    y = y[:,0]
    print(y.size)
    for i in range(y.size):
        print(y[i])
        y[i]=int(y[i].split("Group")[1])
    adata1 = sc.AnnData(x)
    adata1.obs['cell_labels'] = y

    # sc.pp.pca(adata)
    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata)
    # sc.tl.leiden(adata, key_added="clusters")
    adata1 = normalize(adata1, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    x = adata1.X
    y = adata1.obs['cell_labels']

    A, A_norm = load_graph(x)

    x = numpy_to_torch(x).to(opt.args.device)
    A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    model = scDFCN(n_node=x.shape[0]).to(opt.args.device)
    print(model)


    pretrain(model,LoadDataset(x), A_norm)
    adata_embedding,cluster_lables, acc_reuslt,nmi_result,ari_result = train(model, LoadDataset(x), x, y, A, A_norm)
    save(cluster_lables)
    # sc.pp.neighbors(adata_raw, n_neighbors=10, n_pcs=40)
    # sc.tl.umap(adata_raw)
    # plotClusteringImg4static(adata_raw, opt.args.name+'_scDGFC', cluster_lables, 'umap')
    # plotClusteringImg4static(adata_raw, opt.args.name + '_raw', y, 'umap')
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    # print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

    # sc.tl.umap(adata_final)