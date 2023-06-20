from __future__ import print_function, division
from torch import optim
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
from stellargraph.core import StellarGraph
from pandas import Series
from src.util import config
# from src.models import FedLGCN_Plus, LocalLGCN_Plus 
from src.models import feat_loss
import time
from sklearn import preprocessing, model_selection


class LocalOnwer():
    def __init__(self, do_id:int, subG:StellarGraph,
                 num_samples:list, model_path:list):
        
        self.do_id = do_id
        self.subG = subG
        self.num_samples=num_samples # [central_1_hop, central_2_hop]
        
        self.feat_shape = subG.node_features()[0].shape[-1]
        
        self.hide_ids = self.wn_hide_id()
        self.hasG_hide = self.hide_graph()
        self.n_nodes_hide = len(list(self.hasG_hide.nodes()))
        self.num_pred = config.num_pred
        
        self.get_train_test_feat_targets()
        self.neighgen=LocalLGCN_Plus(feat_shape=self.feat_shape, node_len=len(self.all_ids),
                                     node_ids=self.all_ids)
        
        self.model_path=model_path
        
        self.optimizaer = optim.Adam(self.neighgen.parameters(),
                                     lr=config.lr, weight_decay=config.weight_decay)
        
        if config.cuda:
            torch.cuda.empty_cache()
            self.neighgen.cuda()
            self.all_feat = self.all_feat.cuda()
            self.adj = self.adj.cuda()
            self.edges = self.edges.cuda()
            self.all_targets_missing = self.all_targets_missing.cuda()
            self.all_targets_feat = self.all_targets_feat.cuda()
            self.train_ilocs = torch.tensor(self.train_ilocs).cuda()
            self.val_ilocs = torch.tensor(self.train_ilocs).cuda()
            self.test_ilocs = torch.tensor(self.test_ilocs).cuda()
    
    
    def hide_graph(self):
        self.wn_hide_ids=list(set(self.subG.nodes()).difference(self.hide_ids))
        rm_hide_G=self.subG.subgraph(self.wn_hide_ids)
        return rm_hide_G
    
    #def wn_hide_id(self):
        # hide_len=int()
        
    def get_adj(self, edges, node_len):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, -1])),
                            shape=(node_len, node_len),
                            dtype=np.float32)
        
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        if config.cuda:
            adj=adj.cuda()
        return adj
    
    
    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    def get_train_test_feat_targets(self):
        self.all_ids = list(self.hasG_hide.nodes())
        
        self.all_targets_missing=[]
        self.all_targets_feat=[]
        self.all_feat = self.hasG_hide.node_features()
        
        
    
    # def accuracy_missing(self, output)
    
    def greedy_l2_feat(self, pred_feats, true_feats, pred_missing, true_missing):
        pred_feats = pred_feats.view(-1, self.num_pred, self.feat_shape)
        return feat_loss.greedy_loss(pred_feats, true_feats, pred_missing, true_missing).unsqueeze(0).mean().float()
    
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def train_neighgen(self, epoch):
        t = time.time()
        self.neighgen.train()
        self.optimizer.zero_grad()
        input_feat=self.all_feat
        input_edge=self.edges
        input_adj=self.adj
        output_missing,output_feat = self.neighgen(input_feat, input_edge, input_adj)
        output_missing = torch.flatten(output_missing)
        output_feat=output_feat.view(len(self.all_ids),self.num_pred,self.feat_shape)
        self.train_ilocs = self.train_ilocs.long()
        loss_train_missing = F.smooth_l1_loss(output_missing[self.train_ilocs].float(),
                                        self.all_targets_missing[self.train_ilocs].reshape(-1).float())
        
        loss_train_feat = feat_loss.greedy_loss(output_feat[self.train_ilocs],
                                            self.all_targets_feat[self.train_ilocs],
                                            output_missing[self.train_ilocs],
                                            self.all_targets_missing[self.train_ilocs]).unsqueeze(0).mean().float()
        
        acc_train_missing = self.accuracy_missing(output_missing[self.train_ilocs], self.all_targets_missing[self.train_ilocs])
        loss = (config.a * loss_train_missing + config.b * loss_train_feat).float()
        loss.backward()
        
        self.optimizer.step()
        
        self.neighgen.eval()
        val_missing, val_feat = self.neighgen(self.all_feat, self.edges, self.adj)
        val_feat = val_feat.view(len(self.all_ids), self.num_pred, self.feat_shape)
        acc_val_missing = self.accuracy_missing(val_missing[self.train_ilocs], self.all_targets_missing[self.train_ilocs])
        l2_val_feat = self.greedy_l2_feat(val_feat[self.train_ilocs],
                                          self.all_targets_feat[self.train_ilocs],
                                          val_missing[self.train_ilocs],
                                          self.all_targets_missing[self.train_ilocs])
        
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'missing_train: {:.4f}'.format(acc_train_missing),
              'loss_miss: {:.4f}'.format(loss_train_missing.item()),
              'loss_feat: {:.4f}'.format(loss_train_feat.item()),
              'missing_val: {:.4f}'.format(acc_val_missing),
              'l2_val: {:.4f}'.format(l2_val_feat),
              'time: {:.4f}s'.format(time.time() - t))
        
        
    def train(self):
        for epoch in range(config.gen_epochs):
            self.train_neighgen(epoch)
        
        print("NeighGen Finished!")
        
    def save_model(self):
        torch.save(self.neighgen, self.model_path[1])
        
    def load_model(self):
        if config.cuda:
            self.neighgen = torch.load(self.model_path[1], map_location=torch.device('cuda'))
        else:
            self.neighgen = torch.load(self.model_path[1], map_location=torch.device('cpu'))
            
    
    # def set_fed_mode(self):
        