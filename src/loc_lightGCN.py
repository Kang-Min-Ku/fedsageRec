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
from sklearn.model_selection import train_test_split

from src.models.lightgcn import LightGCN
from src.util.eval_implicit import eval_implicit

class loc_LightGCN:
    def __init__(self, dataset, testset):
        # dataset - dataowner
        
        self.dataset = dataset
        self.testset = testset
        self.top_k = config.top_k
        self.n_owners = config.num_owners
        
        precs = []
        recalls = []
        ndcgs = []
        hits = []
        
        self.adj_mat = dataset
        
        for owner_i in range(self.n_owners):
            self.adj_mat[owner_i] = sg.StellarGraph.to_adjacency_matrix(self.dataset[owner_i].hasG)
            #print('adj mat - ' + str(self.adj_mat[owner_i]))
            #print('adj mat type - ' + str(type(self.adj_mat[owner_i])))
        
        for owner_i in range(self.n_owners):
            train, valid = train_test_split(self.adj_mat[owner_i], test_size=0.1, random_state=1234)
            lightGCN = LightGCN(train, valid)
            lightGCN.fit()
            
            #check_param = lightGCN.embedding_dict
            #print('Check Model parameter - ' + str(len(list(check_param['user_emb']))))
            
            #lightgcn_prec, lightgcn_recall, lightgcn_ndcg = eval_implicit(lightGCN, self.adj_mat[owner_i], self.testset, self.top_k)
            lightgcn_prec, lightgcn_recall, lightgcn_ndcg, lightgcn_hit = eval_implicit(lightGCN, train, self.testset, self.top_k)
            precs.append(lightgcn_prec)
            recalls.append(lightgcn_recall)
            ndcgs.append(lightgcn_ndcg)
            hits.append(lightgcn_hit)
            
        avg_prec = sum(precs) / self.n_owners
        avg_recall = sum(recalls) / self.n_owners
        avg_ndcg = sum(ndcgs) / self.n_owners
        avg_hit = sum(hits) / self.n_owners
            
        print(f"Loc LightGCN: prec@{self.top_k} {avg_prec}, recall@{self.top_k} {avg_recall}, ndcg@{self.top_k} {avg_ndcg}, hit@{self.top_k} {avg_hit}")