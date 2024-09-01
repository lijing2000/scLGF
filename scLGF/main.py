from __future__ import print_function, division
from utils import *
import opt
from  scDFCN import scDFCN
from train import train, acc_reuslt, nmi_result, f1_result, ari_result
from PlotClustering import plotClusteringImg,plotClusteringImg4static
from preprocess import normalize
import time
if __name__ == "__main__":
    #
    start_time = time.time()
    print(opt.args.name)
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.args.name =="Macosko":
        data_mat = h5py.File(opt.args.data_file)
        x = np.array(data_mat['X'])
        y = np.array(data_mat['Y'])
        data_mat.close()
        adata = sc.AnnData(x)
        adata.obs['cell_labels'] = y
        cluster_number = int(max(y) - min(y) + 1)
        adata_raw = read_dataset(adata,
                             transpose=False,
                             test_split=False,
                             copy=True)

        adata = normalize(adata_raw, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        x = adata.X
        y = adata.obs['cell_labels']
    else:
        file_path = "data/" + opt.args.name + "." + opt.args.load_type
        dataset = load_data_origin_data(file_path, opt.args.load_type, scaling=True)
        x = dataset.x
        y = dataset.y
        adata = sc.AnnData(x)
        adata.obs['cell_labels'] = y

        x1 = dataset.x1
        y1 = dataset.y1
        adata_raw = sc.AnnData(x1)
        adata_raw.obs['cell_labels'] = y1


    # sc.pp.pca(adata)
    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata)
    # sc.tl.leiden(adata, key_added="clusters")
    # adata_nom = normalize(adata, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True,
    #                   logtrans_input=True)
    # x = adata_nom.X
    # y = adata_nom.obs['cell_labels']

    A, A_norm = load_graph(x)

    x = numpy_to_torch(x).to(opt.args.device)
    A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    model = scDFCN(n_node=x.shape[0]).to(opt.args.device)
    print(model)


    pretrain(model,LoadDataset(x), A_norm)
    adata_embedding,cluster_lables, acc_reuslt,nmi_result,ari_result,gamma_list = train(model, LoadDataset(x), x, y, A, A_norm)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time}秒")

    save(cluster_lables)
    sc.pp.neighbors(adata_raw, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_raw)
    # plotClusteringImg(adata_embedding, opt.args.name+'_scLGF', cluster_lables, 'tsne')
    plotClusteringImg(adata_raw, opt.args.name + '_raw', y, 'tsne')

    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    # print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

    import matplotlib.pyplot as plt

    epochs = range(0, opt.args.epoch)
    plt.plot( gamma_list,acc_reuslt, color=(0, 0, 0), label='beta')  # 也可以用RGB值表示颜色

    plt.show()
    print('finishing training')

    # sc.tl.umap(adata_final)