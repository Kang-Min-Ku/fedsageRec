import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import random
import faiss
import numpy as np
from time import time


class data_value_evaluator(nn.Module):
    def __init__(self):
        super(data_value_evaluator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(64 + 64, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(50 + 1682, 30), nn.ReLU(), nn.Linear(30, 1))
        self.fc4 = nn.Sequential(nn.Linear(50 + 38048, 256), nn.ReLU(), nn.Linear(256, 1))
        #1682, 40981, 38048 = 아이템숫자랑 같음
    def forward(self, user_emb, pos_emb, y_hat_input):
        z = torch.cat([user_emb, pos_emb], dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        cat = torch.cat([z, y_hat_input], dim=1)
        dve = self.fc4(cat)
        return torch.sigmoid(dve).squeeze()

class fedgnn(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(fedgnn, self).__init__()
        self.args = args
        self.n_users = n_user
        self.n_items = n_item
        self.batch_size = args.batch_size
        self.device = args.device

        self.emb_size = 64 #args.embed_size #64
        self.n_layers = 3 #3
        self.ssl_temp = 0.1 #0.1
        self.ssl_reg = 1e-7 #1e-7
        self.hyper_layers = 1 #1
        self.alpha = 1 #1

        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.decay = 1e-4 #self.reg_weight
        self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

        # self.restore_user_e = None
        # self.restore_item_e = None
        self.proto_reg = 8e-8
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.cluster_k = 12 # number of clusters


    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_size)

        initializer(self.user_embedding.weight)
        initializer(self.item_embedding.weight)

    def e_step(self):

        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()

        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.cluster_k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def _convert_sp_mat_to_sp_tensor(self, X):
        #init
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self):

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        embeddings_list = [ego_embeddings]

        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            layer_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            embeddings_list.append(layer_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings) #2층 output
        norm_user_emb2 = F.normalize(previous_user_embeddings) #0층
        norm_all_user_emb = F.normalize(previous_user_embeddings_all) #그냥 다네..
        #pos
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        #neg
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss


    def bpr_loss(self, users_emb, pos_items_emb, neg_items_emb):

        #mf_loss
        pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), axis=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_items_emb), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        #emb_loss
        regularizer = (torch.norm(users_emb) ** 2
                       + torch.norm(pos_items_emb) ** 2
                       + torch.norm(neg_items_emb) ** 2) / 2

        emb_loss = self.decay * regularizer / self.batch_size

        bpr_loss = mf_loss + emb_loss

        return bpr_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        #test
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def calculate_loss(self, users, pos_items, neg_items):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        proto_loss = self.ProtoNCE_loss(center_embedding, users, pos_items)
        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, users, pos_items)

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]

        bpr_loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        return bpr_loss + ssl_loss + proto_loss

    def calculate_2loss(self, users, pos_items, neg_items):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, users, pos_items)

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]

        bpr_loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        return bpr_loss + ssl_loss

    def calculate_bpr_only(self, users, pos_items, neg_items):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]
        bpr_loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        return bpr_loss

    def graph_embedding_expansion(self):
        # 이웃 임베딩 불러오기
        with open('local_user_embs.npy', 'rb') as f:
            neighbor_embs = np.load(f, allow_pickle=True)
        neighor_embs = torch.Tensor(neighbor_embs).to(self.device)
        self.user_embedding = nn.Embedding.from_pretrained(neighor_embs) #내꺼 남에꺼 다들어있음
        # print(self.user_embedding.weight)

    def graph_emb_save(self):

        if os.path.exists(os.path.join(os.getcwd(),'local_user_embs.npy')):
            #load other client embs
            with open('local_user_embs.npy', 'rb') as f:
                neighbor_embs = np.load(f, allow_pickle=True)
        else:
            neighbor_embs = 0.0

        #save local embs
        with open('local_user_embs.npy', 'wb') as f:
            old_embs = copy.deepcopy(self.user_embedding.weight)
            old_embs = old_embs.cpu()
            new_embs = neighbor_embs + 0.1*(old_embs.detach().numpy())

            np.save(f, new_embs, allow_pickle=True)


