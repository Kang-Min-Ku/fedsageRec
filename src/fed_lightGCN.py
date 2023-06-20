import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn, optim, Tensor
from stellargraph.core import StellarGraph
import stellargraph as sg
from src.util import config
import dill as pickle
from scipy.sparse import vstack, coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split

from src.models.lightgcn import LightGCN
from src.util.eval_implicit import eval_implicit

class fed_LightGCN:
    def __init__(self, dataset, testset, final_adj_mat):
        # dataset - dataowner
        
        self.dataset = dataset
        self.testset = testset
        self.top_k = config.top_k
        self.n_owners = config.num_owners
        
        lightGCN_models = []
        param_user_emb = []
        param_item_emb = []
        
        self.adj_mat = dataset
        self.final_adj_mat = final_adj_mat
        
        for owner_i in range(self.n_owners):
            self.adj_mat[owner_i] = sg.StellarGraph.to_adjacency_matrix(self.dataset[owner_i].hasG)
            try:
                self.final_mat += self.adj_mat[owner_i].todense()
            except:
                self.final_mat = self.adj_mat[owner_i].todense()
        
        self.final_mat = csr_matrix(self.final_mat)
        
        for owner_i in range(self.n_owners):
            train, valid = train_test_split(self.adj_mat[owner_i], test_size=0.1, random_state=1234)
            lightGCN = LightGCN(train, valid)
            lightGCN.fit()
            lightGCN_models.append(lightGCN)
            param_user_emb.append(list(lightGCN.embedding_dict['user_emb']))
            param_item_emb.append(list(lightGCN.embedding_dict['item_emb']))
            #print('user emb len - ' + str(len(param_user_emb[owner_i])))
            #print('item emb len - ' + str(len(param_item_emb[owner_i])))
            #print('type - ' + str(type(param_item_emb[owner_i])))
            try:
                self.final_mat_check += train.todense()
            except:
                self.final_mat_check = train.todense()
                
        self.final_mat_check = csr_matrix(self.final_mat_check)
    
        user_emb_len = len(param_user_emb[0])
        item_emb_len = len(param_item_emb[0])
        
        param_user_emb_avg = [sum(values) / user_emb_len for values in zip(*param_user_emb)]
        param_item_emb_avg = [sum(values) / item_emb_len for values in zip(*param_item_emb)]
        
        #print('Final user emb - ' + str(len(param_user_emb_avg)))
        #print('Final item emb - ' + str(len(param_item_emb_avg)))
        
        combined_model = lightGCN_models[0]
        #combined_model.embedding_dict = nn.ParameterDict({
        #    'user_emb': nn.Parameter(torch.tensor(param_user_emb_avg)),
        #    'item_emb': nn.Parameter(torch.tensor(param_item_emb_avg))
        #})
        
        combined_model.state_dict()['user_emb'] = param_user_emb_avg
        combined_model.state_dict()['item_emb'] = param_item_emb_avg
        
        #combined_model.embedding_dict['user_emb'].data = torch.nn.Parameter(torch.tensor(param_user_emb_avg))
        #combined_model.embedding_dict['item_emb'].data = torch.nn.Parameter(torch.tensor(param_item_emb_avg))
        
        lightgcn_prec, lightgcn_recall, lightgcn_ndcg, lightgcn_hit = eval_implicit(combined_model, self.final_mat_check, self.testset, self.top_k)
        print(f"Fed LightGCN: prec@{self.top_k} {lightgcn_prec}, recall@{self.top_k} {lightgcn_recall}, ndcg@{self.top_k} {lightgcn_ndcg}, hit@{self.top_k} {lightgcn_hit}")