import opt
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import *
import tqdm
import opt

class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae

class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

    def forward(self, x):
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # if opt.args.name == "dblp" or opt.args.name == "hhar":
        #     self.act = nn.Tanh()
        #     self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # elif opt.args.name == "usps" or opt.args.name == "acm" or opt.args.name == "cite":
        #     self.act = nn.Tanh()
        #     self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # elif opt.args.name == "reut":
        #     self.act = nn.LeakyReLU(0.2, inplace=True)
        #     self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.act = nn.LeakyReLU(0.01, inplace=True)
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        # if active:
        #     if opt.args.name == "dblp" or opt.args.name == "hhar":
        #         support = self.act(F.linear(features, self.weight))  # add bias
        #     elif opt.args.name == "usps" or opt.args.name == "acm" or opt.args.name == "cite" or opt.args.name == "reut":
        #         support = self.act(torch.mm(features, self.weight))
        # else:
        #     if opt.args.name == "dblp" or opt.args.name == "hhar":
        #         support = F.linear(features, self.weight)   # add bias
        #     elif opt.args.name == "usps" or opt.args.name == "acm" or opt.args.name == "cite" or opt.args.name == "reut":
        #         support = torch.mm(features, self.weight)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        return output


class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=False if opt.args.name == "hhar" else True)
        z = self.gnn_2(z, adj, active=False if opt.args.name == "hhar" else True)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj, active=False if opt.args.name == "hhar" else True)
        z = self.gnn_5(z, adj, active=False if opt.args.name == "hhar" else True)
        z_hat = self.gnn_6(z, adj, active=False if opt.args.name == "hhar" else True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

    def forward(self, x, adj):
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat







class scDFCN(nn.Module):

    def __init__(self, v=1.0, n_node=None, device=None):
        super(scDFCN, self).__init__()

        self.ae = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1,
            ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_enc_3=opt.args.ae_n_enc_3,
            ae_n_dec_1=opt.args.ae_n_dec_1,
            ae_n_dec_2=opt.args.ae_n_dec_2,
            ae_n_dec_3=opt.args.ae_n_dec_3,
            n_input=opt.args.n_input,
            n_z=opt.args.n_z)

        self.gae = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_enc_3=opt.args.gae_n_enc_3,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            gae_n_dec_3=opt.args.gae_n_dec_3,
            n_input=opt.args.n_input)


        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True).to(device)
        self.b = 1 - self.a



        self.cluster_layer = nn.Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + (1 - self.a) * z_igae
        # print(self.a)
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        print(self.gamma)
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde,self.gamma.data.cpu().numpy()




    def add_noise(self, X):
        noise = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
        X_tilde = X * noise
        return X_tilde

    def pretrain(self,dataset,A_norm):
        X = dataset.x
        # X = numpy_to_torch(X).to(opt.args.device)
        X_tilde = self.add_noise(X)
        # optimizer1 = Adam(self.ae.parameters(), lr=opt.args.pre_lr)

        # for epoch in tqdm.tqdm(range(opt.args.pre_epoch_ae)):
        #     # for batch_idx, (x, _) in enumerate(train_loader):
        #
        #     # X_tilde, _ = gaussian_noised_feature(X)
        #     z_ae = self.ae.encoder(X_tilde)
        #     x_hat = self.ae.decoder(z_ae)
        #     loss = F.mse_loss(x_hat, X)
        #
        #     optimizer1.zero_grad()
        #     loss.backward()
        #     optimizer1.step()
        #     print(epoch, loss)
        #
        # optimizer2 = Adam(self.gae.parameters(), lr=opt.args.pre_lr)
        # for epoch in tqdm.tqdm(range(opt.args.pre_epoch_gae)):
        #     # for batch_idx, (x, _) in enumerate(train_loader):
        #
        #     # X_tilde, _ = gaussian_noised_feature(X)
        #     z_igae, z_igae_adj= self.gae.encoder(X_tilde, A_norm)
        #     z_hat, z_hat_adj= self.gae.decoder(z_igae, A_norm)
        #     adj_hat = z_igae_adj + z_hat_adj
        #
        #     loss_w = F.mse_loss(z_hat, torch.spmm(A_norm, X))
        #     loss_a = F.mse_loss(adj_hat, A_norm.to_dense())
        #     loss = loss_w + 0.1 * loss_a
        #
        #     optimizer2.zero_grad()
        #     loss.backward()
        #     optimizer2.step()
        #     print(epoch, loss)

        optimizer3 = Adam(self.parameters(), lr=opt.args.pre_lr)
        for epoch in tqdm.tqdm(range(opt.args.pre_epoch)):
            # for batch_idx, (x, _) in enumerate(train_loader):
            # X_tilde1, X_tilde2 = gaussian_noised_feature(X)
            # X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = self(X_tilde1, A_norm, X_tilde2, A_norm)
            # z_ae = self.ae.encoder(X_tilde)
            # z_igae, z_igae_adj = self.gae.encoder(X_tilde, A_norm)
            # z_i = self.a * z_ae + (1 - self.a) * z_igae
            # z_l = torch.spmm(A_norm, z_i)
            # s = torch.mm(z_l, z_l.t())
            # s = F.softmax(s, dim=1)
            # z_g = torch.mm(s, z_l)
            # z_tilde = self.gamma * z_g + z_l
            # z_hat, z_hat_adj= self.gae.decoder(z_tilde, A_norm)
            # adj_hat = z_igae_adj + z_hat_adj
            x_hat, z_ae = self.ae.forward(X_tilde)
            loss_ae = F.mse_loss(x_hat,X)

            z_igae, z_hat, adj_hat = self.gae.forward(X_tilde, A_norm)

            loss_w = F.mse_loss(z_hat, torch.spmm(A_norm, X))
            loss_a = F.mse_loss(adj_hat, A_norm.to_dense())
            loss = loss_w + loss_a+loss_ae

            optimizer3.zero_grad()
            loss.backward()
            optimizer3.step()
            print(epoch, loss)

        # torch.save(self.state_dict(), 'model_pretrain/{}_pretrain.pkl'.format(opt.args.name))

# class GAE(nn.Module):
#     r"""GAE的代码"""
#     def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
#         super(GAE, self).__init__()
#         self.encoder = GAE_encoder(
#             gae_n_enc_1=gae_n_enc_1,
#             gae_n_enc_2=gae_n_enc_2,
#             gae_n_enc_3=gae_n_enc_3,
#             n_input=n_input)
#
#         self.decoder = InnerProductDecoder()
#         # GAE.reset_parameters(self)
#
#     def forward(self, x, adj):
#         z_igae, z_igae_adj = self.encoder(x, adj)
#         adj_hat = self.decoder(z_igae)
#         return z_igae, adj_hat
#
#
#
#
# class GAE_encoder(nn.Module):
#
#     def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
#         super(GAE_encoder, self).__init__()
#         self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
#         self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
#         self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
#         self.s = nn.Sigmoid()
#
#     def forward(self, x, adj):
#         z = self.gnn_1(x, adj, active=False if opt.args.name == "hhar" else True)
#         z = self.gnn_2(z, adj, active=False if opt.args.name == "hhar" else True)
#         z_igae = self.gnn_3(z, adj, active=False)
#         z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
#         return z_igae, z_igae_adj
#
# class InnerProductDecoder(nn.Module):
#     r"""这内积解码器，即将隐层表示Z内积之后来重建原来的Graph
#     值得注意的，有两个forward可以分别hold住全部重建和只对局部采样重建"""
#     #
#     # def forward(self, z, edge_index, sigmoid=True):
#     #     # 计算节点对之间存在边的概率
#     #     # edge_index分别存的邻接矩阵的行和列，所以取0和1直接可计算
#     #     value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#     #
#     #     # Sigmoid控制是否非线性
#     #     return torch.sigmoid(value) if sigmoid else value
#
#     def forward(self, z, sigmoid=True):
#         # 计算所有节点，所以是按照公式直接内积
#         adj = torch.matmul(z, z.t())
#
#         # Sigmoid控制是否非线性
#         return torch.sigmoid(adj) if sigmoid else adj
#
#
# class scDFCN(nn.Module):
#
#     def __init__(self, v=1.0, n_node=None, device=None):
#         super(scDFCN, self).__init__()
#
#         self.ae = AE(
#             ae_n_enc_1=opt.args.ae_n_enc_1,
#             ae_n_enc_2=opt.args.ae_n_enc_2,
#             ae_n_enc_3=opt.args.ae_n_enc_3,
#             ae_n_dec_1=opt.args.ae_n_dec_1,
#             ae_n_dec_2=opt.args.ae_n_dec_2,
#             ae_n_dec_3=opt.args.ae_n_dec_3,
#             n_input=opt.args.n_input,
#             n_z=opt.args.n_z)
#
#         self.gae = GAE(
#             gae_n_enc_1=opt.args.gae_n_enc_1,
#             gae_n_enc_2=opt.args.gae_n_enc_2,
#             gae_n_enc_3=opt.args.gae_n_enc_3,
#             n_input=opt.args.n_input)
#
#         self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True).to(
#             device)
#         self.b = 1 - self.a
#
#         self.cluster_layer = nn.Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
#         torch.nn.init.xavier_normal_(self.cluster_layer.data)
#
#         self.v = v
#         self.gamma = Parameter(torch.zeros(1))
#
#     def forward(self, x, adj):
#         z_ae = self.ae.encoder(x)
#         z_igae, z_igae_adj = self.gae.encoder(x, adj)
#         z_i = self.a * z_ae + (1 - self.a) * z_igae
#         # print(self.a)
#         z_l = torch.spmm(adj, z_i)
#         s = torch.mm(z_l, z_l.t())
#         s = F.softmax(s, dim=1)
#         z_g = torch.mm(s, z_l)
#         z_tilde = self.gamma * z_g + z_l
#         x_hat = self.ae.decoder(z_tilde)
#         # z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
#         # adj_hat = z_igae_adj + z_hat_adj
#         adj_hat = self.gae.decoder(z_tilde)
#
#         q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#         q = q.pow((self.v + 1.0) / 2.0)
#         q = (q.t() / torch.sum(q, 1)).t()
#
#         q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#         q1 = q1.pow((self.v + 1.0) / 2.0)
#         q1 = (q1.t() / torch.sum(q1, 1)).t()
#
#         q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#         q2 = q2.pow((self.v + 1.0) / 2.0)
#         q2 = (q2.t() / torch.sum(q2, 1)).t()
#
#         return x_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde
#
#         def as_train(self, x, adj):
#             z_ae = self.ae.encoder(x)
#             z_igae, z_igae_adj = self.gae.encoder(x, adj)
#             z_i = self.a * z_ae + (1 - self.a) * z_igae
#             # print(self.a)
#             z_l = torch.spmm(adj, z_i)
#             s = torch.mm(z_l, z_l.t())
#             s = F.softmax(s, dim=1)
#             z_g = torch.mm(s, z_l)
#             z_tilde = self.gamma * z_g + z_l
#             x_hat = self.ae.decoder(z_tilde)
#             z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
#             adj_hat = z_igae_adj + z_hat_adj
#
#             q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#             q = q.pow((self.v + 1.0) / 2.0)
#             q = (q.t() / torch.sum(q, 1)).t()
#
#             q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#             q1 = q1.pow((self.v + 1.0) / 2.0)
#             q1 = (q1.t() / torch.sum(q1, 1)).t()
#
#             q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#             q2 = q2.pow((self.v + 1.0) / 2.0)
#             q2 = (q2.t() / torch.sum(q2, 1)).t()
#
#             return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde


