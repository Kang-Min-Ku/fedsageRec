import torch
import torch.nn as nn
import torch.nn.functional as F

class data_value_evaluator(nn.Module):
    def __init__(self):
        super(data_value_evaluator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(64 + 64, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(50 + 400, 30), nn.ReLU(), nn.Linear(30, 1))
        self.fc4 = nn.Sequential(nn.Linear(50 + 1682, 30), nn.ReLU(), nn.Linear(30, 1)) #ml-100k
        #400, 1682자리 = 아이템숫자랑 같음
    def forward(self, user_emb, pos_emb, y_hat_input):
        z = torch.cat([user_emb, pos_emb], dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        cat = torch.cat([z, y_hat_input], dim=1)
        dve = self.fc4(cat)
        return torch.sigmoid(dve).squeeze()

class TwolossNCL(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(TwolossNCL, self).__init__()
        self.args = args
        self.n_users = n_user
        self.n_items = n_item
        self.batch_size = args.batch_size

        self.device = args.device

        self.emb_size = 64 #64
        self.n_layers = 3 #3
        self.ssl_temp = 0.1 #0.1
        self.ssl_reg = 1e-7
        self.hyper_layers = 1 #1
        self.alpha = 1 #1

        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.decay = 1e-5 #self.reg_weight

        self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

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

        #검증필요 특히 self.변수이름 안맞음
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
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
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