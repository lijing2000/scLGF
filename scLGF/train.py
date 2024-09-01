import tqdm

import opt
from utils import *
from torch.optim import Adam
import torch.nn.functional as F
import scanpy as sc


acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
tol=1e-3
gamma_list = []


def train(model, dataset, x, y, A, A_norm):
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    original_acc = opt.args.acc


    print("Training on {}…".format(opt.args.name))

    # model_init(model, dataset, x, y, A_norm)
    # pretrained_dict = torch.load('model_pretrain/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde,_ = model(x, A_norm)
    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(opt.args.device)
    eva(y, cluster_id, 'Initialization')
    # sadj = A_norm.cpu().numpy()

    y_pred_last = cluster_id
    num = x.shape[0]



    for epoch in range(opt.args.epoch):
        # if opt.args.name in use_adjust_lr:
        #     adjust_learning_rate(optimizer, epoch)
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde ,gamma = model(x, A_norm)
        gamma_list.append(gamma)
        tmp_q = q.data
        p = target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, x)
        loss_w = F.mse_loss(z_hat, torch.spmm(A_norm, x))
        loss_a = F.mse_loss(adj_hat, A_norm.to_dense())
        loss_igae =loss_w +opt.args.gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + opt.args.lambda_value *loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('{} loss: {}'.format(epoch, loss))


        # kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
        kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
        labels = kmeans.labels_
        # from sklearn.cluster import SpectralClustering
        # labels = SpectralClustering(n_clusters=opt.args.n_clusters, affinity="precomputed", assign_labels="discretize",
        #                             random_state=0).fit_predict(z_tilde.data.cpu().numpy())
        acc, nmi, ari= eva(y, labels, epoch)
        # if y is not None:
        #     acc = np.round(cluster_acc(y, kmeans.labels_), 5)
        #     y = list(map(int, y))
        #     kmeans.labels_ = np.array(kmeans.labels_)
        #     nmi = np.round(metrics.normalized_mutual_info_score(y, kmeans.labels_), 5)
        #     ari = np.round(metrics.adjusted_rand_score(y, kmeans.labels_), 5)
        #     print('ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        print("epoch:",epoch," acc:",acc," nmi:",nmi," ari:",ari)
        # for parameters in model.parameters():  # 打印出参数矩阵及值
        #     print(parameters)
        #
        # for name, parameters in model.named_parameters():  # 打印出每一层的参数的大小
        #     print(name, ':', parameters.size())
        #
        # for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())

        delta_label = np.sum(labels != y_pred_last).astype(np.float32) / num
        if epoch > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print("Reach tolerance threshold. Stopping training.")
            break

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if acc > original_acc:
        #     original_acc = acc
            # torch.save(model.state_dict(), 'model_final/{}_final.pkl'.format(opt.args.name))
    # adata = sc.AnnData(z_tilde.cpu().detach().numpy())
    adata = sc.AnnData(z_tilde.cpu().detach().numpy())
    cluster_lables=kmeans.labels_
    adata_x = sc.AnnData(x_hat.cpu().detach().numpy())
    adata_x.obs['cell_labels'] = cluster_lables

    return adata,cluster_lables, acc_reuslt,nmi_result,ari_result,gamma_list




