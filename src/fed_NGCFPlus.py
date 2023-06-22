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
import torch.nn as nn
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split

from src.models import feat_loss
from src.models import models
from src.util import mending_graph
from scipy.sparse import vstack, coo_matrix, csr_matrix
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.loc_lightGCNPlus import LocalOwner
from src.util.eval_implicit import eval_implicit
import src.util.normalize_matrix as normalize_matrix

"""
all_targets_missing: missing nodes
all_targets_feat: missing nodes의 feature
"""

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
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # #print(output_feat, output_missing)
            # print(output_feat[0], output_missing[0],
            #       local_owners[i].all_targets_missing[local_owners[i].train_ilocs].reshape(-1).float()[0],
            #       local_owners[i].all_targets_feat[local_owners[i].train_ilocs][0])
            # print(output_feat.size(), output_missing.size(),
            #       local_owners[i].all_targets_missing[local_owners[i].train_ilocs].reshape(-1).float().size(),
            #       local_owners[i].all_targets_feat[local_owners[i].train_ilocs].size())
            # print(np.unique(output_missing.detach().cpu().numpy()))
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
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
                    # print("###################################################")
                    # #print(global_target_feat, local_owners[i].all_targets_feat[local_owners[i].train_ilocs])
                    # print(len(global_target_feat), local_owners[i].all_targets_feat[local_owners[i].train_ilocs].size())
                    # print("###################################################")
                    # exit(1)
                    loss += config.b * loss_train_feat_other
            loss = 1.0 / config.num_owners * loss
            loss.backward()
            optim_list[i].step()

    return local_owners


class fed_NGCFPlus:
    def __init__(self, dataset, testset, global_train_adj, num_users, num_items, encoder_args):
        self.dataset = dataset
        self.testset = testset
        self.top_k = config.top_k
        self.n_owners = config.num_owners
        self.global_train_adj = global_train_adj # neighgen causes final_mat in fed_lightGCNplus.py -> just test by global train matrix

        global_weights = {}
        local_encoders = [] # NGCF model 
        local_datasets = []
        
        #prepare local model & dataset
        for owner_i in range(self.n_owners):
            dataset_i = {}
            adj_mat = StellarGraph.to_adjacency_matrix(self.dataset[owner_i].subG)
            dataset_i["adj"] = adj_mat
            norm_adj_mat = normalize_matrix.mean_adj_single(adj_mat.todok()) # mean_adj_mat
            dataset_i["norm_adj"] = norm_adj_mat

            train, valid = train_test_split(adj_mat, test_size=config.classifier_valid_ratio, random_state=1234)
            dataset_i["train"] = train
            dataset_i["valid"] = valid
            local_datasets.append(dataset_i)

            encoder = NGCF(num_users, num_items, norm_adj_mat, dataset_i, encoder_args)
            local_encoders.append(encoder)

        for c_round in range(config.communication_round):
            for owner_i in range(self.n_owners):
                local_encoders[owner_i].fit()
                aggregated_weights = self.aggregate(local_encoders, config.aggregate_exception)
                # 여기부터
    
    # simple fedavg
    # is this code allow backpropagation?
    def aggregate(self, local_models, exception:list):
        weights = {}
        for m in local_models:
            for name, param in m.named_parameters():
                skip = not all([not name.endswith(e) for e in exception])
                if skip:
                    continue

                try:
                    weights[name] = weights[name] + param
                except:
                    weights[name] = param

        for name, param in weights.items():
            weights[name] = nn.Parameter(param / self.n_owners)

        return weights
               
    def distribute(self, exception:list):
        with torch.no_grad():
            for m in self.local_models:
                for name, param in m.named_parameters():
                    skip = not all([not name.endswith(e) for e in exception])
                    if skip:
                        continue

                    param.copy_(self.global_weights[name] / self.n_owners) #안합쳐져요 ㅅㅂ