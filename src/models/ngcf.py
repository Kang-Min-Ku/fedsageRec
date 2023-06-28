import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.util import config
from scipy.sparse import vstack, coo_matrix, csr_matrix
from src.util.eval_implicit import eval_implicit, eval_implicit_NGCF

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, dataset, norm_adj):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        #self.device = config.device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_size = config.latent_dim
        self.batch_size = config.batch_size
        self.node_dropout = config.dropout
        self.mess_dropout = config.mess_dropout

        self.norm_adj = norm_adj

        self.layers = config.layers
        self.reg = config.regs
        self.bpr_decay = config.bpr_decay
        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

        self.optimizer = config.optimizer(self.parameters(), lr=config.lr, weight_decay=self.reg)
        self.dataset = dataset

        self.to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers #임베딩 [64] + 레이어 [64,64,64]
        for k in range(len(self.layers)): #w1, w2 두개
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col]) #0이 아닌 값이 있는 위치
        v = torch.from_numpy(coo.data).float() #0이 아닌 값들
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape): #rate에 node dropout 들어감=0.1, noise shape은 0이 아닌 갯수..
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] #boolean

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate)) #숫자를 크게만들기... 그래디언트가 작게되니까 시그널 크기..올려줌....

    def create_bpr_loss(self, users, pos_items, neg_items): #bayesian personalized ranking
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer 같음
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        emb_loss = self.bpr_decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t()) #matmul은 벡터와 메트릭스 자유롭게 계산

    def forward(self, users, pos_items, neg_items, drop_flag=False):
        #dropout한 adjacency matrix 엣지 유무
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj #nnz는 행렬에 포함된 0이 아닌 요소의 갯수

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)  #이웃정보 가져옴 = ei

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k] #웨이트곱함 W1

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings) #원소별곱셈 그냥 아무것도 안한 자기 자신과 이웃정보 받은 인풋이랑...
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k] #W2

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings) #더한 값을 활성화 함수에 넣어주고

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings) #드랍아웃 함번 해주고

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings] #각층마다 나온 결과 모아모아

        all_embeddings = torch.cat(all_embeddings, 1) #다 컨캣해버려 여프 = 피쳐의 종류가 늘어난 거지
        u_g_embeddings = all_embeddings[:self.n_user, :] #위에는 유저꺼~
        i_g_embeddings = all_embeddings[self.n_user:, :] #밑엥는 아이템꺼~

        """
        *********************************************************
        look up.
        """
        #print('i_g_embeddings - ' + str(len(i_g_embeddings)))
        #print('pos items - ' + str(pos_items))
        
        #print('user embeddings - ' + str(len(u_g_embeddings))) 
        
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]
        
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
    
    def train_model_per_batch(self, users, pos_items, neg_items, users_in_owner):
        self.optimizer.zero_grad()
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.forward(users, pos_items, neg_items)
        batch_loss, batch_mf_loss, batch_emb_loss = self.create_bpr_loss(u_g_embeddings,
                                                                         pos_i_g_embeddings,
                                                                         neg_i_g_embeddings,
                                                                        )
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss
    
    def fit(self):
        user_idx = np.arange(self.n_user)
        patience = 0
        best_score = 0.

        for epoch in range(config.epochs_local):
            epoch_loss = 0.
            self.train()

            np.random.RandomState(1234).shuffle(user_idx)
            batch_num = len(user_idx) // self.batch_size + 1

            for batch_idx in range(batch_num):
                try:
                    batch_users = user_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                except:
                    batch_users = user_idx[batch_idx*self.batch_size:]

                batch_users, batch_pos_items, batch_neg_items, users_in_owner= \
                    self.sample_pair(batch_users) #Some users are empty because of test setting
                batch_loss = self.train_model_per_batch(batch_users, batch_pos_items, batch_neg_items, users_in_owner)

                if torch.isnan(batch_loss):
                    print('Loss NaN. Train finish.')
                    break
                
                epoch_loss += batch_loss

            '''
            if epoch % config.test_every == 0:
                with torch.no_grad():
                    self.eval()
                    top_k = config.top_k
                    #error!
                    ndcg, prec, recall = eval_implicit_NGCF(self, self, self.dataset["train"], self.dataset["valid"], top_k)

                    if epoch % config.print_every == 0:
                        print("[NGCF] epoch %d, loss: %f"%(epoch, epoch_loss))
                        print(f"(NGCF) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}")
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

            if config.use_early_stopping and patience > config.early_stopping_patience:
                #print(f"Early Stopping at epoch {epoch}")
                break     
                '''           
    
    def sample_pair(self, users):
        csr_adj_mat = csr_matrix(self.norm_adj)
        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            # pos_items = self.train_items[u]
            pos_items = csr_adj_mat[u].nonzero()[1]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id] - self.n_user

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_item,size=1)[0]
                if neg_id not in csr_adj_mat[u].nonzero()[1] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
        
        sampled_users, pos_items, neg_items = [], [], []
        for u in users:
            try:
                new_pos = sample_pos_items_for_u(u, 1)
                sampled_users.append(u)
            except:
                new_pos = [0]
            new_neg = sample_neg_items_for_u(u, 1)
            # try:
            #     new_pos = sample_neg_items_for_u(u, 1)
            #     new_neg = sample_neg_items_for_u(u, 1)
            #     sampled_users.append(u)
            # except:
            #     new_pos = 0
            #     new_neg = 0
            pos_items += new_pos
            neg_items += new_neg
            # pos_items += sample_pos_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items, sampled_users
        #return users, pos_items, neg_items

    def predict(self, user_ids, item_ids):        
        with torch.no_grad():
            u_g_embeddings, pos_embeddings, _ = self.forward(user_ids, 0, 0)
            scores = self.rating(u_g_embeddings, pos_embeddings)
            predict_ = scores.detach().cpu().numpy()
            
            #print('predict len - ' + str(len(predict_)))
            #print('item_ids len - ' + str(len(item_ids)))
            
            return predict_