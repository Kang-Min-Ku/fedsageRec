import copy
import sys
sys.path.append("..") # I don't like it
from util import annealing
import os

import numpy as np
import torch.optim as optim
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F

class data_value_evaluator(nn.Module):
    def __init__(self, args):
        super(data_value_evaluator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(64 + 64, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(50 + 1682, 30), nn.ReLU(), nn.Linear(30, 1))
        num_items = 0
        if args.dataset == "ml-100k":
            num_items = 1682
        elif args.dataset == "gowalla":
            num_items = 40981
        elif args.dataset == "yelp2018":
            num_items = 38048
        elif args.dataset == "amazon-book":
            num_items = 91649 - 50
        self.fc4 = nn.Sequential(nn.Linear(50 + num_items, 256), nn.ReLU(), nn.Linear(256, 1))
        #1682, 40981, 38048 = 아이템숫자랑 같음
    def forward(self, user_emb, pos_emb, y_hat_input):
        z = torch.cat([user_emb, pos_emb], dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        cat = torch.cat([z, y_hat_input], dim=1)
        dve = self.fc4(cat)
        return torch.sigmoid(dve).squeeze()

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.args = args
        self.n_users = n_user
        self.n_items = n_item
        self.device = args.device
        self.bpr_reg = args.bpr_reg

        self.emb_size = args.embed_size # args.embed_size #64
        self.n_layers = args.n_layers
        self.batch_size = args.batch_size
        self.node_dropout = eval(args.node_dropout)[0]
        self.mess_dropout = eval(args.mess_dropout)

        # hyper-parameter for loss
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg #1e-7
        self.alpha = args.alpha
        self.hyper_layers = args.hyper_layers

        self.proto_reg = args.proto_reg #8e-8
        # self.user_centroids = None
        # self.user_2cluster = None
        # self.item_centroids = None
        # self.item_2cluster = None

        if self.args.annealing_function == "linear":
            self.annealing_func = annealing.linear
        elif self.args.annealing_function == "sigmoid":
            self.annealing_func = annealing.sigmoid
        elif self.args.annealing_function == "exponential_increasing":
            self.annealing_func = annealing.exponential_increasing
        elif self.args.annealing_function == "exponential_decreasing":
            self.annealing_func = annealing.exponential_decreasing

        self.cluster_k = args.cluster_k

        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.decay = args.decay
        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))})
        weight_dict = nn.ParameterDict()

        layers = [self.emb_size] + self.layers # 64 + 64,64,64
        for k in range(len(self.layers)): # w1, w2
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        #init
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape): #rate 0.1
        #train
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask] #boolean
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate)) # 시그널 크기 확대

    def create_bpr_loss(self, users, pos_items, neg_items): #bayesian personalized ranking
        #Test
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer 같음
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss # , mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        #test
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self):
        drop_flag = True
        A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embedding = [ego_embeddings]
        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #이웃정보 가져옴 = ei
            # W1
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]
            # W2
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embedding += [norm_embeddings]
        ## concat
        # all_embeddings = torch.cat(all_embedding, 1)
        ## weighted sum
        all_embeddings = torch.stack(all_embedding[:self.n_layers + 1], dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]
        # u_g_embeddings = u_g_embeddings[users, :]
        # pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        # neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, i_g_embeddings, all_embedding

    def get_positive_centroid(self, embeddings, centroids):
        # embeddings = embeddings.detach().cpu().numpy()
        # torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))

        sims = torch.matmul(embeddings, centroids.transpose(0, 1))
        max_sim_idx = torch.argmax(sims, dim=1)
        embs2centroid = centroids[max_sim_idx]

        return embs2centroid

    def get_cluster_data(self, users, pos_items, neg_items, client_user_centroid):
        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]

        user_embeddings = u_embeddings.detach().cpu().numpy()
        item_embeddings = pos_embeddings.detach().cpu().numpy()

        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        # grobal item centroid 만들면..? 밑에 loss에서 사용가능..? nope.. 일단 만들어
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

        client_user_centroid = torch.Tensor(client_user_centroid).to(self.args.device)

        # 비교해서 데이터 골라서 return
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # similarity = [cos(client_user_centroid, i) for i in self.user_centroids]
        similarity = [torch.matmul(client_user_centroid, i) for i in self.user_centroids]

        selected_cluster = similarity.index(max(similarity))
        selected_user = []

        for i, user in enumerate(self.user_2cluster):
            if user == selected_cluster:
                selected_user.append(i)
        # global data...중에 저 user 골라냅시다~
        cluster_data = {}
        n_data = 0
        for i in selected_user:
            cluster_data[i] = self.args.big_data[i]
            n_data += len(cluster_data[i])

        return cluster_data, n_data, self.user_centroids, self.item_centroids

    def get_one_centroid(self, users, pos_items, neg_items):
        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        # neg_embeddings = item_all_embeddings[neg_items]

        user_embeddings = u_embeddings.detach().cpu().numpy()
        pos_item_embeddings = pos_embeddings.detach().cpu().numpy()
        user_centroid = np.mean(user_embeddings, axis=0)
        # print("centroid ???", len(user_centroid))

        return user_centroid

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

    def calculate_loss(self, users, pos_items, neg_items, g_user_centroids, g_item_centroids):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]

        center_embedding = embeddings_list[0]
        cl_loss = self.clustering_loss(center_embedding, users, pos_items, g_user_centroids, g_item_centroids)
        bpr_loss = self.create_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        if self.args.use_CL:
            return cl_loss + (self.bpr_reg * bpr_loss)
        else:
            return self.bpr_reg * bpr_loss

        #return cl_loss + ( self.bpr_reg * bpr_loss )
    def structure_loss(self, users, pos_items, user, item):

        # users, pos_items = 전체
        # user, item = 선택된

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
        previous_embedding = embeddings_list[0]
        one_hop_embedding = embeddings_list[1]
        current_embedding = embeddings_list[2]

        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])
        ##############################
        # ###### 추가한 부분 user
        selected_user_embeddings = current_user_embeddings[user] #current user embedding: 전체 embedding => ngcf를 통과한 final embedding 
        selected_pre_user_embeddings = previous_user_embeddings_all[user]
        norm_selected_users = F.normalize(selected_user_embeddings)  # 이웃
        norm_selected_user = F.normalize(selected_pre_user_embeddings)  # 0층
        # ### items
        selected_item_embeddings = current_item_embeddings[item]
        selected_pre_item_embeddings = previous_item_embeddings_all[item]
        norm_selected_items = F.normalize(selected_item_embeddings)  # 이웃
        norm_selected_item = F.normalize(selected_pre_item_embeddings)  # 0층
        ##############################
        """
        문제점: len(users) != len(user) -> 오류 => line 304에서 오류남
        """
        current_user_embeddings = current_user_embeddings[users]
        previous_user_embeddings = previous_user_embeddings_all[users]
        norm_user_emb1 = F.normalize(current_user_embeddings) #2층 output
        norm_user_emb2 = F.normalize(previous_user_embeddings) #0층
        norm_all_user_emb = F.normalize(previous_user_embeddings_all) #그냥 다네..
        #pos
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        # before gdve 추가!!!! 어떻게? pos item 1홉 이웃 값 평균과 나!! 새로운 이웃에 대한 정의임
        #users_t = torch.Tensor(users).to(self.device)
        #user_t = torch.Tensor(user).to(self.device)
        pos_score_plus = torch.mul(norm_selected_users, norm_selected_user).sum(dim=1)
        #user_gdve_plus_idx = torch.where(users_t.unsqueeze(0) == user_t.unsqueeze(1))[1] #index of members of user in users
        user_gdve_plus_idx = [users.index(u) for u in user]
        #pos_score_user[user_gdve_plus_idx] = 0.9*pos_score_user[user_gdve_plus_idx] + 0.1*pos_score_plus
        sl_new_user_score_ratio = self.args.sl_new_user_score_ratio
        if not self.args.sl_use_user_gdve_plus:
            sl_new_user_score_ratio = 0.0
        # pos_score_user[user_gdve_plus_idx] = (1.0-sl_new_user_score_ratio)*pos_score_user[user_gdve_plus_idx] + \
        #                     sl_new_user_score_ratio*pos_score_plus
        pos_score_user[user_gdve_plus_idx] = pos_score_user[user_gdve_plus_idx] + \
                            sl_new_user_score_ratio*pos_score_plus
        #pos_score_user = pos_score_user + 0.1*pos_score_plus #torch.mul(norm_selected_users, norm_selected_user).sum(dim=1)
        #neg
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[pos_items]
        previous_item_embeddings = previous_item_embeddings_all[pos_items]
        norm_item_emb1 = F.normalize(current_item_embeddings) #2층
        norm_item_emb2 = F.normalize(previous_item_embeddings) #0층
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        #pos
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        # 추가
        item_gdve_plus_idx = [pos_items.index(i) for i in item]
        #pos_score_item[item_gdve_plus_idx] = pos_score_item[item_gdve_plus_idx] + 0.1*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
        
        sl_new_item_score_ratio = self.args.sl_new_item_score_ratio
        if not self.args.sl_use_item_gdve_plus:
            sl_new_item_score_ratio = 0.0
        # pos_score_item[item_gdve_plus_idx] = (1.0-sl_new_item_score_ratio)*pos_score_item[item_gdve_plus_idx] + \
        #                     sl_new_item_score_ratio*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
        pos_score_item[item_gdve_plus_idx] = pos_score_item[item_gdve_plus_idx] + \
                            sl_new_item_score_ratio*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
        #pos_score_item = pos_score_item + 0.1*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
        #neg
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        annealing_coeff = 1.0
        curr_round = int(os.environ[self.args.comm_round_environ_key])
        max_round = 0
        if self.args.anneal_by_max_round:
            max_round = self.args.comm_round
        if self.args.use_SL_annealing:
            annealing_coeff = self.annealing_func(curr_round,
                                                  self.args.SL_temperature,
                                                  self.args.first_annealing_round,
                                                  max_round)

        ssl_loss = annealing_coeff * self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)

        #ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    # def structure_loss(self, user, item):
    #     # user, item = selected by gdve
    #     user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
    #     previous_embedding = embeddings_list[0]
    #     one_hop_embedding = embeddings_list[1]
    #     current_embedding = embeddings_list[2]

    #     current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
    #     previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
    #                                                                              [self.n_users, self.n_items])
    #     # 추가한 부분
    #     middle_user_embeddings, middle_item_embeddings = torch.split(one_hop_embedding, [self.n_users, self.n_items])
    #     selected_user = middle_item_embeddings[item]
    #     norm_new_user = F.normalize(selected_user)
    #     selected_item = middle_user_embeddings[user]
    #     norm_new_item = F.normalize(selected_item)

    #     current_user_embeddings = current_user_embeddings[user]
    #     previous_user_embeddings = previous_user_embeddings_all[user]
    #     norm_user_emb1 = F.normalize(current_user_embeddings) #2층 output
    #     norm_user_emb2 = F.normalize(previous_user_embeddings) #0층
    #     norm_all_user_emb = F.normalize(previous_user_embeddings_all) #그냥 다네..
    #     #pos
    #     pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
    #     # before gdve 추가!!!! 어떻게? pos item 1홉 이웃 값 평균과 나!! 새로운 이웃에 대한 정의임
    #     sl_new_user_score_ratio = self.args.sl_new_user_score_ratio
    #     if not self.args.sl_use_user_gdve_plus:
    #         sl_new_user_score_ratio = 0.0
    #     pos_score_user = (1.0-sl_new_user_score_ratio)*pos_score_user + \
    #                         sl_new_user_score_ratio*torch.mul(norm_new_user, norm_user_emb2).sum(dim=1)
        
    #     # pos_score_user = pos_score_user + 0.1*torch.mul(norm_new_user, norm_user_emb2).sum(dim=1)
    #     #neg
    #     ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
    #     pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
    #     ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

    #     ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

    #     current_item_embeddings = current_item_embeddings[item]
    #     previous_item_embeddings = previous_item_embeddings_all[item]
    #     norm_item_emb1 = F.normalize(current_item_embeddings) #2층
    #     norm_item_emb2 = F.normalize(previous_item_embeddings) #0층
    #     norm_all_item_emb = F.normalize(previous_item_embeddings_all)
    #     #pos
    #     pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
    #     sl_new_item_score_ratio = self.args.sl_new_item_score_ratio
    #     if not self.args.sl_use_item_gdve_plus:
    #         sl_new_item_score_ratio = 0.0
    #     pos_score_item = (1.0-sl_new_item_score_ratio)*pos_score_item + \
    #                         sl_new_item_score_ratio*torch.mul(norm_new_item, norm_item_emb2).sum(dim=1)
    #     # pos_score_item = pos_score_item + 0.1*torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
    #     #neg
    #     ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
    #     pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
    #     ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

    #     ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

    #     annealing_coeff = 1.0
    #     curr_round = int(os.environ[self.args.comm_round_environ_key])
    #     max_round = 0
    #     if self.args.anneal_by_max_round:
    #         max_round = self.args.comm_round
    #     if self.args.use_SL_annealing:
    #         annealing_coeff = self.annealing_func(curr_round,
    #                                               self.args.SL_temperature,
    #                                               self.args.first_annealing_round,
    #                                               max_round)

    #     ssl_loss = annealing_coeff * self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
    #     return ssl_loss

    def clustering_loss(self, node_embedding, user, item, g_user_centroids, g_item_centroids):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)
        # user별 가장 유사한 centroid list 찾는 함수
        user2centroids = self.get_positive_centroid(norm_user_embeddings, g_user_centroids)

        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, g_user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)
        # 새로 추가
        item2centroids = self.get_positive_centroid(norm_item_embeddings, g_item_centroids)
        # item2cluster = self.item_2cluster[item]  # [B, ]
        # item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, g_item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        annealing_coeff = 1.0
        curr_round = int(os.environ[self.args.comm_round_environ_key])
        max_round = 0
        if self.args.anneal_by_max_round:
            max_round = self.args.comm_round
        if self.args.use_CL_annealing:
            annealing_coeff = self.annealing_func(curr_round,
                                                  self.args.SL_temperature,
                                                  self.args.first_annealing_round,
                                                  max_round)

        proto_nce_loss = annealing_coeff * self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def calculate_bpr_loss(self, users, pos_items, neg_items):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]

        annealing_coeff = 1.0
        curr_round = int(os.environ[self.args.comm_round_environ_key])
        max_round = 0
        if self.args.anneal_by_max_round:
            max_round = self.args.comm_round
        if self.args.use_BPR_annealing:
            annealing_coeff = self.annealing_func(curr_round,
                                                  self.args.SL_temperature,
                                                  self.args.first_annealing_round,
                                                  max_round)        

        bpr_loss = annealing_coeff * self.create_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        return bpr_loss


    # original ngcf
    # def forward(self, users, pos_items, neg_items, drop_flag=True):
    #
    #     A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
    #                                 self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
    #     ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
    #                                 self.embedding_dict['item_emb']], 0)
    #     all_embeddings = [ego_embeddings]
    #
    #     for k in range(len(self.layers)):
    #         side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #이웃정보 가져옴 = ei
    #         # W1
    #         sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
    #                          + self.weight_dict['b_gc_%d' % k]
    #         # W2
    #         bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
    #         bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
    #                                         + self.weight_dict['b_bi_%d' % k]
    #         ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
    #         ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
    #         # normalize the distribution of embeddings.
    #         norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
    #         all_embeddings += [norm_embeddings]
    #
    #     all_embeddings = torch.cat(all_embeddings, 1)
    #     u_g_embeddings = all_embeddings[:self.n_users, :]
    #     i_g_embeddings = all_embeddings[self.n_users:, :]
    #     u_g_embeddings = u_g_embeddings[users, :]
    #     pos_i_g_embeddings = i_g_embeddings[pos_items, :]
    #     neg_i_g_embeddings = i_g_embeddings[neg_items, :]
    #
    #     return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings


    # original ngcf
    # def forward(self, users, pos_items, neg_items, drop_flag=True):
    #
    #     A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
    #                                 self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
    #     ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
    #                                 self.embedding_dict['item_emb']], 0)
    #     all_embeddings = [ego_embeddings]
    #
    #     for k in range(len(self.layers)):
    #         side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #이웃정보 가져옴 = ei
    #         # W1
    #         sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
    #                          + self.weight_dict['b_gc_%d' % k]
    #         # W2
    #         bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
    #         bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
    #                                         + self.weight_dict['b_bi_%d' % k]
    #         ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
    #         ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
    #         # normalize the distribution of embeddings.
    #         norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
    #         all_embeddings += [norm_embeddings]
    #
    #     all_embeddings = torch.cat(all_embeddings, 1)
    #     u_g_embeddings = all_embeddings[:self.n_users, :]
    #     i_g_embeddings = all_embeddings[self.n_users:, :]
    #     u_g_embeddings = u_g_embeddings[users, :]
    #     pos_i_g_embeddings = i_g_embeddings[pos_items, :]
    #     neg_i_g_embeddings = i_g_embeddings[neg_items, :]
    #
    #     return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

# import copy
# import sys
# sys.path.append("..") # I don't like it
# from util import annealing
# import os

# import numpy as np
# import torch.optim as optim
# import faiss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class data_value_evaluator(nn.Module):
#     def __init__(self, args):
#         super(data_value_evaluator, self).__init__()
#         self.fc1 = nn.Sequential(nn.Linear(64 + 64, 100), nn.ReLU())
#         self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
#         self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())
#         # self.fc4 = nn.Sequential(nn.Linear(50 + 1682, 30), nn.ReLU(), nn.Linear(30, 1))
#         num_items = 0
#         if args.dataset == "ml-100k":
#             num_items = 1682
#         elif args.dataset == "gowalla":
#             num_items = 40981
#         elif args.dataset == "yelp2018":
#             num_items = 38048
#         self.fc4 = nn.Sequential(nn.Linear(50 + num_items, 256), nn.ReLU(), nn.Linear(256, 1))
#         #1682, 40981, 38048 = 아이템숫자랑 같음
#     def forward(self, user_emb, pos_emb, y_hat_input):
#         z = torch.cat([user_emb, pos_emb], dim=1)
#         z = self.fc1(z)
#         z = self.fc2(z)
#         z = self.fc3(z)
#         cat = torch.cat([z, y_hat_input], dim=1)
#         dve = self.fc4(cat)
#         return torch.sigmoid(dve).squeeze()

# class NGCF(nn.Module):
#     def __init__(self, n_user, n_item, norm_adj, args):
#         super(NGCF, self).__init__()
#         self.args = args
#         self.n_users = n_user
#         self.n_items = n_item
#         self.device = args.device
#         self.bpr_reg = args.bpr_reg

#         self.emb_size = args.embed_size # args.embed_size #64
#         self.n_layers = args.n_layers
#         self.batch_size = args.batch_size
#         self.node_dropout = eval(args.node_dropout)[0]
#         self.mess_dropout = eval(args.mess_dropout)

#         # hyper-parameter for loss
#         self.ssl_temp = args.ssl_temp
#         self.ssl_reg = args.ssl_reg #1e-7
#         self.alpha = args.alpha
#         self.hyper_layers = args.hyper_layers

#         self.proto_reg = args.proto_reg #8e-8
#         # self.user_centroids = None
#         # self.user_2cluster = None
#         # self.item_centroids = None
#         # self.item_2cluster = None

#         if self.args.annealing_function == "linear":
#             self.annealing_func = annealing.linear
#         elif self.args.annealing_function == "sigmoid":
#             self.annealing_func = annealing.sigmoid
#         elif self.args.annealing_function == "exponential_increasing":
#             self.annealing_func = annealing.exponential_increasing
#         elif self.args.annealing_function == "exponential_decreasing":
#             self.annealing_func = annealing.exponential_decreasing

#         self.cluster_k = args.cluster_k

#         self.norm_adj = norm_adj
#         self.layers = eval(args.layer_size)
#         self.decay = args.decay
#         self.embedding_dict, self.weight_dict = self.init_weight()
#         self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

#     def init_weight(self):
#         # xavier init
#         initializer = nn.init.xavier_uniform_

#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))})
#         weight_dict = nn.ParameterDict()

#         layers = [self.emb_size] + self.layers # 64 + 64,64,64
#         for k in range(len(self.layers)): # w1, w2
#             weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
#             weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
#             weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
#             weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

#         return embedding_dict, weight_dict

#     def _convert_sp_mat_to_sp_tensor(self, X):
#         #init
#         coo = X.tocoo()
#         i = torch.LongTensor([coo.row, coo.col])
#         v = torch.from_numpy(coo.data).float()
#         return torch.sparse.FloatTensor(i, v, coo.shape)

#     def sparse_dropout(self, x, rate, noise_shape): #rate 0.1
#         #train
#         random_tensor = 1 - rate
#         random_tensor += torch.rand(noise_shape)
#         dropout_mask = torch.floor(random_tensor).type(torch.bool)
#         i = x._indices()
#         v = x._values()
#         i = i[:, dropout_mask]
#         v = v[dropout_mask] #boolean
#         out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
#         return out * (1. / (1 - rate)) # 시그널 크기 확대

#     def create_bpr_loss(self, users, pos_items, neg_items): #bayesian personalized ranking
#         #Test
#         pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
#         neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
#         maxi = nn.LogSigmoid()(pos_scores - neg_scores)
#         mf_loss = -1 * torch.mean(maxi)

#         # cul regularizer 같음
#         regularizer = (torch.norm(users) ** 2
#                        + torch.norm(pos_items) ** 2
#                        + torch.norm(neg_items) ** 2) / 2

#         emb_loss = self.decay * regularizer / self.batch_size

#         return mf_loss + emb_loss # , mf_loss, emb_loss

#     def rating(self, u_g_embeddings, pos_i_g_embeddings):
#         #test
#         return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

#     def forward(self):
#         drop_flag = True
#         A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
#                                     self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
#         ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
#                                     self.embedding_dict['item_emb']], 0)
#         all_embedding = [ego_embeddings]
#         for k in range(len(self.layers)):
#             side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #이웃정보 가져옴 = ei
#             # W1
#             sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
#                              + self.weight_dict['b_gc_%d' % k]
#             # W2
#             bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
#             bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
#                                             + self.weight_dict['b_bi_%d' % k]
#             ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
#             ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
#             # normalize the distribution of embeddings.
#             norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
#             all_embedding += [norm_embeddings]
#         ## concat
#         # all_embeddings = torch.cat(all_embedding, 1)
#         ## weighted sum
#         all_embeddings = torch.stack(all_embedding[:self.n_layers + 1], dim=1)
#         all_embeddings = torch.mean(all_embeddings, dim=1)

#         u_g_embeddings = all_embeddings[:self.n_users, :]
#         i_g_embeddings = all_embeddings[self.n_users:, :]
#         # u_g_embeddings = u_g_embeddings[users, :]
#         # pos_i_g_embeddings = i_g_embeddings[pos_items, :]
#         # neg_i_g_embeddings = i_g_embeddings[neg_items, :]

#         return u_g_embeddings, i_g_embeddings, all_embedding

#     def get_positive_centroid(self, embeddings, centroids):
#         # embeddings = embeddings.detach().cpu().numpy()
#         # torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))

#         sims = torch.matmul(embeddings, centroids.transpose(0, 1))
#         max_sim_idx = torch.argmax(sims, dim=1)
#         embs2centroid = centroids[max_sim_idx]

#         return embs2centroid

#     def get_cluster_data(self, users, pos_items, neg_items, client_user_centroid):
#         user_all_embeddings, item_all_embeddings, _ = self.forward()
#         u_embeddings = user_all_embeddings[users]
#         pos_embeddings = item_all_embeddings[pos_items]

#         user_embeddings = u_embeddings.detach().cpu().numpy()
#         item_embeddings = pos_embeddings.detach().cpu().numpy()

#         self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
#         # grobal item centroid 만들면..? 밑에 loss에서 사용가능..? nope.. 일단 만들어
#         self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

#         client_user_centroid = torch.Tensor(client_user_centroid).to(self.args.device)

#         # 비교해서 데이터 골라서 return
#         # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#         # similarity = [cos(client_user_centroid, i) for i in self.user_centroids]
#         similarity = [torch.matmul(client_user_centroid, i) for i in self.user_centroids]

#         selected_cluster = similarity.index(max(similarity))
#         selected_user = []

#         for i, user in enumerate(self.user_2cluster):
#             if user == selected_cluster:
#                 selected_user.append(i)
#         # global data...중에 저 user 골라냅시다~
#         cluster_data = {}
#         n_data = 0
#         for i in selected_user:
#             cluster_data[i] = self.args.big_data[i]
#             n_data += len(cluster_data[i])

#         return cluster_data, n_data, self.user_centroids, self.item_centroids

#     def get_one_centroid(self, users, pos_items, neg_items):
#         user_all_embeddings, item_all_embeddings, _ = self.forward()

#         u_embeddings = user_all_embeddings[users]
#         pos_embeddings = item_all_embeddings[pos_items]
#         # neg_embeddings = item_all_embeddings[neg_items]

#         user_embeddings = u_embeddings.detach().cpu().numpy()
#         pos_item_embeddings = pos_embeddings.detach().cpu().numpy()
#         user_centroid = np.mean(user_embeddings, axis=0)
#         # print("centroid ???", len(user_centroid))

#         return user_centroid

#     def run_kmeans(self, x):
#         """Run K-means algorithm to get k clusters of the input tensor x
#         """
#         kmeans = faiss.Kmeans(d=self.emb_size, k=self.cluster_k, gpu=True)
#         kmeans.train(x)
#         cluster_cents = kmeans.centroids

#         _, I = kmeans.index.search(x, 1)

#         # convert to cuda Tensors for broadcast
#         centroids = torch.Tensor(cluster_cents).to(self.device)
#         centroids = F.normalize(centroids, p=2, dim=1)

#         node2cluster = torch.LongTensor(I).squeeze().to(self.device)
#         return centroids, node2cluster

#     def calculate_loss(self, users, pos_items, neg_items, g_user_centroids, g_item_centroids):

#         user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

#         u_embeddings = user_all_embeddings[users]
#         pos_embeddings = item_all_embeddings[pos_items]
#         neg_embeddings = item_all_embeddings[neg_items]

#         center_embedding = embeddings_list[0]
#         cl_loss = self.clustering_loss(center_embedding, users, pos_items, g_user_centroids, g_item_centroids)
#         bpr_loss = self.create_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

#         if self.args.use_CL:
#             return cl_loss + (self.bpr_reg * bpr_loss)
#         else:
#             return self.bpr_reg * bpr_loss

#         #return cl_loss + ( self.bpr_reg * bpr_loss )
#     def structure_loss(self, users, pos_items, user, item):

#         # users, pos_items = 전체
#         # user, item = 선택된

#         user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
#         previous_embedding = embeddings_list[0]
#         one_hop_embedding = embeddings_list[1]
#         current_embedding = embeddings_list[2]

#         current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
#         previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
#                                                                                  [self.n_users, self.n_items])
#         ##############################
#         # ###### 추가한 부분 user
#         selected_user_embeddings = current_user_embeddings[user] #current user embedding: 전체 embedding => ngcf를 통과한 final embedding 
#         selected_pre_user_embeddings = previous_user_embeddings_all[user]
#         norm_selected_users = F.normalize(selected_user_embeddings)  # 이웃
#         norm_selected_user = F.normalize(selected_pre_user_embeddings)  # 0층
#         # ### items
#         selected_item_embeddings = current_item_embeddings[item]
#         selected_pre_item_embeddings = previous_item_embeddings_all[item]
#         norm_selected_items = F.normalize(selected_item_embeddings)  # 이웃
#         norm_selected_item = F.normalize(selected_pre_item_embeddings)  # 0층
#         ##############################
#         """
#         문제점: len(users) != len(user) -> 오류 => line 304에서 오류남
#         """
#         current_user_embeddings = current_user_embeddings[users]
#         previous_user_embeddings = previous_user_embeddings_all[users]
#         norm_user_emb1 = F.normalize(current_user_embeddings) #2층 output
#         norm_user_emb2 = F.normalize(previous_user_embeddings) #0층
#         norm_all_user_emb = F.normalize(previous_user_embeddings_all) #그냥 다네..
#         #pos
#         pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
#         # before gdve 추가!!!! 어떻게? pos item 1홉 이웃 값 평균과 나!! 새로운 이웃에 대한 정의임
#         #users_t = torch.Tensor(users).to(self.device)
#         #user_t = torch.Tensor(user).to(self.device)
#         pos_score_plus = torch.mul(norm_selected_users, norm_selected_user).sum(dim=1)
#         #user_gdve_plus_idx = torch.where(users_t.unsqueeze(0) == user_t.unsqueeze(1))[1] #index of members of user in users
#         user_gdve_plus_idx = [users.index(u) for u in user]
#         pos_score_user[user_gdve_plus_idx] = 0.9*pos_score_user[user_gdve_plus_idx] + 0.1*pos_score_plus
#         #pos_score_user = pos_score_user + 0.1*pos_score_plus #torch.mul(norm_selected_users, norm_selected_user).sum(dim=1)
#         #neg
#         ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
#         pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
#         ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

#         ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

#         current_item_embeddings = current_item_embeddings[pos_items]
#         previous_item_embeddings = previous_item_embeddings_all[pos_items]
#         norm_item_emb1 = F.normalize(current_item_embeddings) #2층
#         norm_item_emb2 = F.normalize(previous_item_embeddings) #0층
#         norm_all_item_emb = F.normalize(previous_item_embeddings_all)
#         #pos
#         pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
#         # 추가
#         item_gdve_plus_idx = [pos_items.index(i) for i in item]
#         pos_score_item[item_gdve_plus_idx] = pos_score_item[item_gdve_plus_idx] + 0.1*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
#         #pos_score_item = pos_score_item + 0.1*torch.mul(norm_selected_items, norm_selected_item).sum(dim=1)
#         #neg
#         ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
#         pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
#         ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

#         ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

#         annealing_coeff = 1.0
#         curr_round = int(os.environ[self.args.comm_round_environ_key])
#         max_round = 0
#         if self.args.anneal_by_max_round:
#             max_round = self.args.comm_round
#         if self.args.use_SL_annealing:
#             annealing_coeff = self.annealing_func(curr_round,
#                                                   self.args.SL_temperature,
#                                                   self.args.first_annealing_round,
#                                                   max_round)

#         ssl_loss = annealing_coeff * self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)

#         #ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
#         return ssl_loss

#     # def structure_loss(self, user, item):
#     #     # user, item = selected by gdve
#     #     user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
#     #     previous_embedding = embeddings_list[0]
#     #     one_hop_embedding = embeddings_list[1]
#     #     current_embedding = embeddings_list[2]

#     #     current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
#     #     previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
#     #                                                                              [self.n_users, self.n_items])
#     #     # 추가한 부분
#     #     middle_user_embeddings, middle_item_embeddings = torch.split(one_hop_embedding, [self.n_users, self.n_items])
#     #     selected_user = middle_item_embeddings[item]
#     #     norm_new_user = F.normalize(selected_user)
#     #     selected_item = middle_user_embeddings[user]
#     #     norm_new_item = F.normalize(selected_item)

#     #     current_user_embeddings = current_user_embeddings[user]
#     #     previous_user_embeddings = previous_user_embeddings_all[user]
#     #     norm_user_emb1 = F.normalize(current_user_embeddings) #2층 output
#     #     norm_user_emb2 = F.normalize(previous_user_embeddings) #0층
#     #     norm_all_user_emb = F.normalize(previous_user_embeddings_all) #그냥 다네..
#     #     #pos
#     #     pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
#     #     # before gdve 추가!!!! 어떻게? pos item 1홉 이웃 값 평균과 나!! 새로운 이웃에 대한 정의임
#     #     sl_new_user_score_ratio = self.args.sl_new_user_score_ratio
#     #     if not self.args.sl_use_user_gdve_plus:
#     #         sl_new_user_score_ratio = 0.0
#     #     pos_score_user = (1.0-sl_new_user_score_ratio)*pos_score_user + \
#     #                         sl_new_user_score_ratio*torch.mul(norm_new_user, norm_user_emb2).sum(dim=1)
        
#     #     # pos_score_user = pos_score_user + 0.1*torch.mul(norm_new_user, norm_user_emb2).sum(dim=1)
#     #     #neg
#     #     ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
#     #     pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
#     #     ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

#     #     ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

#     #     current_item_embeddings = current_item_embeddings[item]
#     #     previous_item_embeddings = previous_item_embeddings_all[item]
#     #     norm_item_emb1 = F.normalize(current_item_embeddings) #2층
#     #     norm_item_emb2 = F.normalize(previous_item_embeddings) #0층
#     #     norm_all_item_emb = F.normalize(previous_item_embeddings_all)
#     #     #pos
#     #     pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
#     #     sl_new_item_score_ratio = self.args.sl_new_item_score_ratio
#     #     if not self.args.sl_use_item_gdve_plus:
#     #         sl_new_item_score_ratio = 0.0
#     #     pos_score_item = (1.0-sl_new_item_score_ratio)*pos_score_item + \
#     #                         sl_new_item_score_ratio*torch.mul(norm_new_item, norm_item_emb2).sum(dim=1)
#     #     # pos_score_item = pos_score_item + 0.1*torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
#     #     #neg
#     #     ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
#     #     pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
#     #     ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

#     #     ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

#     #     annealing_coeff = 1.0
#     #     curr_round = int(os.environ[self.args.comm_round_environ_key])
#     #     max_round = 0
#     #     if self.args.anneal_by_max_round:
#     #         max_round = self.args.comm_round
#     #     if self.args.use_SL_annealing:
#     #         annealing_coeff = self.annealing_func(curr_round,
#     #                                               self.args.SL_temperature,
#     #                                               self.args.first_annealing_round,
#     #                                               max_round)

#     #     ssl_loss = annealing_coeff * self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
#     #     return ssl_loss

#     def clustering_loss(self, node_embedding, user, item, g_user_centroids, g_item_centroids):
#         user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

#         user_embeddings = user_embeddings_all[user]  # [B, e]
#         norm_user_embeddings = F.normalize(user_embeddings)
#         # user별 가장 유사한 centroid list 찾는 함수
#         user2centroids = self.get_positive_centroid(norm_user_embeddings, g_user_centroids)

#         pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
#         pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
#         ttl_score_user = torch.matmul(norm_user_embeddings, g_user_centroids.transpose(0, 1))
#         ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

#         proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

#         item_embeddings = item_embeddings_all[item]
#         norm_item_embeddings = F.normalize(item_embeddings)
#         # 새로 추가
#         item2centroids = self.get_positive_centroid(norm_item_embeddings, g_item_centroids)
#         # item2cluster = self.item_2cluster[item]  # [B, ]
#         # item2centroids = self.item_centroids[item2cluster]  # [B, e]
#         pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
#         pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
#         ttl_score_item = torch.matmul(norm_item_embeddings, g_item_centroids.transpose(0, 1))
#         ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
#         proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

#         annealing_coeff = 1.0
#         curr_round = int(os.environ[self.args.comm_round_environ_key])
#         max_round = 0
#         if self.args.anneal_by_max_round:
#             max_round = self.args.comm_round
#         if self.args.use_CL_annealing:
#             annealing_coeff = self.annealing_func(curr_round,
#                                                   self.args.SL_temperature,
#                                                   self.args.first_annealing_round,
#                                                   max_round)

#         proto_nce_loss = annealing_coeff * self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
#         return proto_nce_loss

#     def calculate_bpr_loss(self, users, pos_items, neg_items):

#         user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

#         u_embeddings = user_all_embeddings[users]
#         pos_embeddings = item_all_embeddings[pos_items]
#         neg_embeddings = item_all_embeddings[neg_items]

#         annealing_coeff = 1.0
#         curr_round = int(os.environ[self.args.comm_round_environ_key])
#         max_round = 0
#         if self.args.anneal_by_max_round:
#             max_round = self.args.comm_round
#         if self.args.use_BPR_annealing:
#             annealing_coeff = self.annealing_func(curr_round,
#                                                   self.args.SL_temperature,
#                                                   self.args.first_annealing_round,
#                                                   max_round)        

#         bpr_loss = annealing_coeff * self.create_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

#         return bpr_loss


#     # original ngcf
#     # def forward(self, users, pos_items, neg_items, drop_flag=True):
#     #
#     #     A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
#     #                                 self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
#     #     ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
#     #                                 self.embedding_dict['item_emb']], 0)
#     #     all_embeddings = [ego_embeddings]
#     #
#     #     for k in range(len(self.layers)):
#     #         side_embeddings = torch.sparse.mm(A_hat, ego_embeddings) #이웃정보 가져옴 = ei
#     #         # W1
#     #         sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
#     #                          + self.weight_dict['b_gc_%d' % k]
#     #         # W2
#     #         bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
#     #         bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
#     #                                         + self.weight_dict['b_bi_%d' % k]
#     #         ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
#     #         ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
#     #         # normalize the distribution of embeddings.
#     #         norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
#     #         all_embeddings += [norm_embeddings]
#     #
#     #     all_embeddings = torch.cat(all_embeddings, 1)
#     #     u_g_embeddings = all_embeddings[:self.n_users, :]
#     #     i_g_embeddings = all_embeddings[self.n_users:, :]
#     #     u_g_embeddings = u_g_embeddings[users, :]
#     #     pos_i_g_embeddings = i_g_embeddings[pos_items, :]
#     #     neg_i_g_embeddings = i_g_embeddings[neg_items, :]
#     #
#     #     return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings