import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import numpy as np
import scipy.sparse as sp
from IPython import embed
from src.util.eval_implicit import eval_implicit
from src.util import config

# print(torch.__version__)

class LightGCN(nn.Module):
    def __init__(self, train, test):
        super(LightGCN, self).__init__()
        
        self.train_data = train
        self.test_data = test
        
        self.num_users, self.num_items = self.train_data.shape
        
        self.R = sp.csr_matrix(self.train_data)
        
        self.norm_adj = self.create_adj_mat()
        
        self.learning_rate = config.lr
        self.decay = config.weight_decay
        self.batch_size = config.batch_size
        self.num_epochs = config.epoch_recommendation
        self.num_layers = config.n_layers
        self.node_dropout = config.dropout
        #self.device = config.cuda
        #self.device = 'cpu'
        # GPU 시용 nvidia-smi로 확인
        #self.device = torch.device('cuda')
        self.device = config.device
        
        self.emb_size = config.latent_dim
        
        self.embedding_dict = self.init_weight()
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.to(self.device)
        
        
    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items, self.emb_size)))
        })
        
        #embedding_dict = nn.ParameterDict({
        #    'user_emb': nn.Parameter(initializer(torch.empty(self.num_users, self.emb_size))),
        #    'item_emb': nn.Parameter(initializer(ttorch.empty(self.num_items, self.emb_size)))
        #})
        
        return embedding_dict
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        
        return out * (1. / (1 - rate))
    
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    
    def forward(self, users, pos_items, neg_items, drop_flag=False):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        
        for k in range(self.num_layers):
            norm_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings.append(norm_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, 1)
        final_embeddings = torch.mean(all_embeddings, 1)
        
        u_g_embeddings = final_embeddings[:self.num_users, :]
        i_g_embeddings = final_embeddings[self.num_users:, :]
        
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]
        
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, i_g_embeddings
    
    def fit(self):
        user_idx = np.arange(self.num_users)
        patience = 0
        best_score = 0.
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            self.train()
            
            np.random.RandomState(12345).shuffle(user_idx)
            
            batch_num = int(len(user_idx) / self.batch_size) + 1
            
            for batch_idx in range(batch_num):
                batch_users = user_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                batch_matrix = torch.FloatTensor(self.train_data[batch_users, :].toarray()).to(self.device)
                batch_users = torch.LongTensor(batch_users).to(self.device)
                batch_loss = self.train_model_per_batch(batch_matrix, batch_users)
                
                if torch.isnan(batch_loss):
                    print('Loss NaN. Train finish.')
                    break
                
                epoch_loss += batch_loss
            
             
            if epoch % 50 == 0:
                with torch.no_grad():
                    self.eval()
                    
                    top_k = config.top_k
                    print("[LightGCN] epoch %d, loss: %f"%(epoch, epoch_loss))
                    
                    prec, recall, ndcg, hit = eval_implicit(self, self.train_data, self.test_data, top_k)
                    print(f"(LightGCN) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}, hit@{top_k} {hit}")
                    self.train()

                    if config.early_stopping_policy == "recall":
                        if recall > best_score:
                            best_score = recall
                            patience = 0
                        else:
                            patience += 1
                    elif config.early_stopping_policy == "ndcg":
                        if ndcg > best_score:
                            best_score = ndcg
                            patience = 0
                        else:
                            patience += 1
                    elif config.early_stopping_policy == "hit":
                        if hit > best_score:
                            best_score = hit
                            patience = 0
                        else:
                            patience += 1
            if config.use_early_stopping and patience > config.early_stopping_patience:
                print("Early Stopping")
                break
                    
    def train_model_per_batch(self, train_matrix, batch_users, pos_items=0, neg_items=0):
        self.optimizer.zero_grad()
        
        u_g_embeddings, _, _, i_g_embeddings = self.forward(batch_users, 0, 0)
        
        output = self.rating(u_g_embeddings, i_g_embeddings)
        
        loss = F.binary_cross_entropy(torch.sigmoid(output), train_matrix, reduction="none").sum(1).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            u_g_embeddings, _, _, i_g_embeddings = self.forward(user_ids, 0, 0)
            
            output = self.rating(u_g_embeddings, i_g_embeddings)
            
            predict_ = output.detach().cpu().numpy()
            return predict_[item_ids]
        
    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)

        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R).tolil()

        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        '''
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocsr()

        '''
        
        return adj_mat.tocsr()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))