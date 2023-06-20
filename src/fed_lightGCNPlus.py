
from tqdm import tqdm
import pandas as pd
from pandas import Series
import numpy as np
import torch
from torch import optim
from stellargraph.core import StellarGraph
import scipy.sparse as sp
from src.util import config
import dill as pickle
import torch.nn.functional as F
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split

from src.models import feat_loss
from src.models import models
from src.util import mending_graph
from scipy.sparse import vstack, coo_matrix, csr_matrix
from src.models.lightgcn import LightGCN
from src.loc_lightGCNPlus import LocalOwner
from src.util.eval_implicit import eval_implicit

def train_fedgen(local_owners:list, feat_shape:int):
    
    assert len(local_owners) == config.num_owners
    for owner in local_owners:
        assert owner.__class__.__name__ == LocalOwner.__name__
        
    local_gen_list = []
    optim_list = []
    
    for local_i in local_owners:
        local_i.set_fed_model()
        local_gen_list.append(local_i.fed_model.gen)
        optim_list.append(optim.Adam(local_gen_list[-1].parameters(),
                                lr=config.lr, weight_decay=config.weight_decay))
        
    for epoch in range(config.gen_epochs):
        for i in range(config.num_owners):
            local_gen_list[i].train()
            optim_list[i].zero_grad()
            local_model = local_owners[i].fed_model
            input_feat = local_owners[i].all_feat
            input_edge = local_owners[i].edges
            input_adj = local_owners[i].adj
            output_missing, output_feat = local_model(input_feat, input_edge, input_adj)
            output_missing = torch.flatten(output_missing)
            output_feat = output_feat.view(len(local_owners[i].all_ids), local_owners[i].num_pred, local_owners[i].feat_shape)
            
            loss_train_missing = F.smooth_l1_loss(output_missing[local_owners[i].train_ilocs].float(),
                                                    local_owners[i].all_targets_missing[local_owners[i].
                                                    train_ilocs].reshape(-1).float())
            
            loss_train_feat = feat_loss.greedy_loss(output_feat[local_owners[i].train_ilocs],
                                                local_owners[i].all_targets_feat[local_owners[i].train_ilocs],
                                                output_missing[local_owners[i].train_ilocs],
                                                local_owners[i].all_targets_missing[
                                                    local_owners[i].train_ilocs
                                                ]).unsqueeze(0).mean().float()
            
            acc_train_missing = local_owners[i].accuracy_missing(output_missing[local_owners[i].train_ilocs],
                                                        local_owners[i].all_targets_missing[local_owners[i].train_ilocs])
            
            loss = (config.a * loss_train_missing + config.b * loss_train_feat).float()
            '''
            print('Data onwer ' + str(i),
                  ' Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'missing_train: {:.4f}'.format(acc_train_missing),
                  'loss_miss: {:.4f}'.format(loss_train_missing.item()),
                  'loss_feat: {:.4f}'.format(loss_train_feat.item()))
            '''
            
            for j in range(config.num_owners):
                if j != i:
                    choice = np.random.choice(len(list(local_owners[j].subG.nodes())),
                                              len(local_owners[i].train_ilocs))
                    others_ids = local_owners[j].subG.nodes()[choice]
                    global_target_feat = []
                    for c_i in others_ids:
                        neighbors_ids = local_owners[j].subG.neighbors(c_i)
                        while len(neighbors_ids)==0:
                            c_i = np.random.choice(len(list(local_owners[j].subG.nodes())),1)[0]
                            id_i = local_owners[j].subG.nodes()[c_i]
                            neighbors_ids = local_owners[j].subG.neighbors(id_i)
                        choice_i = np.random.choice(neighbors_ids, config.num_pred)
                        for ch_i in choice_i:
                            global_target_feat.append(local_owners[j].subG.node_features([ch_i])[0])
                    global_target_feat = np.asarray(global_target_feat).reshape(
                        (len(local_owners[i].train_ilocs), config.num_pred, feat_shape))
                    loss_train_feat_other = feat_loss.greedy_loss(output_feat[local_owners[i].train_ilocs],
                                                            global_target_feat,
                                                            output_missing[local_owners[i].train_ilocs],
                                                            local_owners[i].all_targets_missing[
                                                                local_owners[i].train_ilocs]
                                                            ).unsqueeze(0).mean().float()
                    loss += config.b * loss_train_feat_other
            loss = 1.0 / config.num_owners * loss
            loss.backward()
            optim_list[i].step()

    return local_owners
            
        
class fed_LightGCNPlus:
    def __init__(self, dataset, testset, final_adj_mat):
        # dataset - dataowner list
        
        #print('data type - ' + str(type(dataset)))
        #print('dataset type - ' + str(type(dataset[0])))
        #print('dataset check - ' + str(type(dataset[0].subG)))
        
        self.dataset = dataset
        self.testset = testset
        self.top_k = config.top_k
        self.n_owners = config.num_owners
        
        lightGCN_models = []
        param_user_emb = []
        param_item_emb = []
        
        self.adj_mat = dataset
        # self.final_mat = 1
        
        for owner_i in range(self.n_owners):
            self.adj_mat[owner_i] = StellarGraph.to_adjacency_matrix(self.dataset[owner_i].subG)
            try:
                self.final_mat += self.adj_mat[owner_i].todense()
            except:
                self.final_mat = self.adj_mat[owner_i].todense()
        
        self.final_mat = csr_matrix(self.final_mat)
        #print('check - ' + str(type(self.final_mat)))
            
        for owner_i in range(self.n_owners):
            train, valid = train_test_split(self.adj_mat[owner_i], test_size=0.1, random_state=1234)
            lightGCN = LightGCN(train, valid)
            lightGCN.fit()
            lightGCN_models.append(lightGCN)
            param_user_emb.append(list(lightGCN.embedding_dict['user_emb']))
            param_item_emb.append(list(lightGCN.embedding_dict['item_emb']))
            
            try:
                self.final_mat_check += train.todense()
            except:
                self.final_mat_check = train.todense()
                
        self.final_mat_check = csr_matrix(self.final_mat_check)
            
        user_emb_len = len(param_user_emb[0])
        item_emb_len = len(param_item_emb[0])
        
        param_user_emb_avg = [sum(values) / user_emb_len for values in zip(*param_user_emb)]
        param_item_emb_avg = [sum(values) / item_emb_len for values in zip(*param_item_emb)]
        
        combined_model = lightGCN_models[0]
        
        combined_model.state_dict()['user_emb'] = param_user_emb_avg
        combined_model.state_dict()['item_emb'] = param_item_emb_avg
        
        lightgcn_prec, lightgcn_recall, lightgcn_ndcg, lightgcn_hit = eval_implicit(lightGCN, self.final_mat, self.testset, self.top_k)
        
        
        print(f"Fed LightGCN Plus: prec@{self.top_k} {lightgcn_prec}, recall@{self.top_k} {lightgcn_recall}, ndcg@{self.top_k} {lightgcn_ndcg}, hit@{self.top_k} {lightgcn_hit}")