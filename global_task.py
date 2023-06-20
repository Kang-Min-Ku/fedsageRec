import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn, optim, Tensor
from stellargraph.core import StellarGraph
from src.util import config
import dill as pickle
from sklearn.model_selection import train_test_split

from src.models.lightgcn import LightGCN
from src.util.eval_implicit import eval_implicit

# Global Graph에 대해서 LightGCN 진행

class Global:
    def __init__(self, globalG, testset):
        # globalG - adj_mat
        # test - dictionary
        
        self.globalG = globalG
        self.testset = testset
        self.top_k = config.top_k
        
        self.trainset, self.validset = train_test_split(self.globalG, test_size=0.1, random_state=1234)
        
        # print('train len - ' + str(self.trainset))
        # print('valid len - ' + str(self.validset))
        
        lightgcn = LightGCN(self.trainset, self.validset)
        lightgcn.fit()
        
        #lightgcn_prec, lightgcn_recall, lightgcn_ndcg = eval_implicit(lightgcn, self.globalG, self.testset, self.top_k)
        lightgcn_prec, lightgcn_recall, lightgcn_ndcg, lightgcn_hit = eval_implicit(lightgcn, self.trainset, self.testset, self.top_k)
        print(f"Global LightGCN: prec@{self.top_k} {lightgcn_prec}, recall@{self.top_k} {lightgcn_recall}, ndcg@{self.top_k} {lightgcn_ndcg}, hitrate@{self.top_k} {lightgcn_hit}")
        
        '''
        self.G = globalG
        self.n_users = n_users
        self.n_items = n_items
        self.testset = testset
        self.n_layers = config.n_layers
        self.latent_dim = config.latent_dim
        self.info_path = config.server_info_dir
        self.test_acc_path = config.global_test_acc_file
        self.globLGCN=None
        
        self.tensorG = self.csr_to_tensor(self.G)
        self.tensorTest = self.csr_to_tensor(self.testset)
        
        # print('tensor G - ' + str(self.tensorG))
        # print('tensor T - ' + str(self.tensorTest))
        
        lightgcn = LightGCN(self.G, self.n_users, self.n_items, self.n_layers, self.latent_dim)
        print("Size of Learnable Embedding : ", list(lightgcn.parameters())[0].size())
        '''
        
         
    def csr_to_tensor(self, data):
        coo = data.tocoo()
        
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
        

    def save_server_info(self):
        if os.path.isfile(self.info_path):
            return
        pickle.dump(self, open(self.info_path, 'wb'))
        return