import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import random
import base64
import numpy as np

class MF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(MF, self).__init__()
        self.args = args
        self.n_users = n_user
        self.n_items = n_item
        self.decay = 1e-5
        self.batch_size = args.batch_size

        self.device = args.device
        self.emb_size = args.embed_size  # 64

        self.norm_adj = norm_adj

        self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        print("training MF start")

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_size)

        initializer(self.user_embedding.weight)
        initializer(self.item_embedding.weight)

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

        user_all_embeddings, item_all_embeddings = torch.split(ego_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, embeddings_list

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        #test
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def calculate_loss(self, users, pos_items, neg_items):
        #rmse
        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]
        loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # rmse
        # label = self.sparse_norm_adj.to_dense() #2625, 2625
        # label = label[:943, 943:]
        # pred = torch.matmul(u_embeddings, i_embeddings.t()) #943,1682
        #
        # error = torch.square(torch.sub(label,pred))
        # loss = torch.sqrt(torch.mean(error))

        return loss

    def bpr_loss(self, users_emb, pos_items_emb, neg_items_emb):

        #mf_loss
        pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), axis=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_items_emb), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)
        # emb_loss
        regularizer = (torch.norm(users_emb) ** 2
                       + torch.norm(pos_items_emb) ** 2
                       + torch.norm(neg_items_emb) ** 2) / 2

        emb_loss = self.decay * regularizer / self.batch_size

        bpr_loss = mf_loss + emb_loss

        return bpr_loss


