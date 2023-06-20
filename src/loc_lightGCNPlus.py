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
from src.models.lightgcn import LightGCN
from src.models.models import FedLightGCN_Plus
from src.util.eval_implicit import eval_implicit

class LocalOwner:
    def __init__(self, do_id:int, subG:StellarGraph):
        self.do_id = do_id,
        self.subG = subG
        self.n_samples = config.num_samples
        
        self.feat_shape = subG.node_features()[0].shape[-1]
        self.num_pred = config.num_pred
        
        self.all_node_index = np.array(subG.nodes())
        self.train_node_index, self.test_node_index = model_selection.train_test_split(
            self.all_node_index, train_size=0.8, test_size=0.1, random_state=1234
        )
        
        self.wn_hide_ids, self.hide_ids = self.wn_hide_index()
        self.hasG_hide = self.hide_graph()
        self.n_nodes_hide = len(list(self.hasG_hide.nodes()))
        
        #print('hasG hide - ' + str(self.hasG_hide))
        #print('n nodes hide - ' + str(self.n_nodes_hide))
        
        self.get_train_test_feat_targets()
        self.neighgen= models.LocalLGCN_Plus(feat_shape=self.feat_shape, node_len=len(self.all_ids),
                                             node_ids=self.all_ids)

        self.optimizer = optim.Adam(self.neighgen.parameters(),
                                    lr=config.lr, weight_decay=config.weight_decay)        
        
        if config.cuda:
            torch.cuda.empty_cache()
            self.neighgen.cuda()
            self.all_feat = self.all_feat.cuda()
            self.adj = self.adj.cuda()
            self.edges=self.edges.cuda()
            self.all_targets_missing = self.all_targets_missing.cuda()
            self.all_targets_feat = self.all_targets_feat.cuda()
            self.train_ilocs = torch.tensor(self.train_ilocs).cuda()
            self.val_ilocs = self.train_ilocs.clone().detach()
            self.test_ilocs = torch.tensor(self.test_ilocs).cuda()

            
    def hide_graph(self):
        self.wn_hide_ids = list(set(self.subG.nodes()).difference(self.hide_ids))
        rm_hide_G = self.subG.subgraph(self.wn_hide_ids)
        return rm_hide_G
    
    def wn_hide_index(self):
        hide_len = int((len(self.all_node_index) - 
                       len(self.train_node_index) - 
                       len(self.test_node_index)) * config.hidden_portion)
        could_hide_ids = np.setdiff1d(np.setdiff1d(self.all_node_index, self.train_node_index),
                                      self.test_node_index)
        
        #print('hide len - ' + str(hide_len))
        #print('colud hide ids  - ' + str(could_hide_ids))
        
        hide_ids = np.random.choice(could_hide_ids, hide_len, replace=False)
        return np.setdiff1d(np.setdiff1d(np.setdiff1d(self.all_node_index, hide_ids),
                            self.train_node_index), self.test_node_index), hide_ids

    def get_adj(self, edges, node_len):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape = (node_len, node_len),
                            dtype=np.float32)
        
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        if config.cuda:
            adj=adj.cuda()
        return adj
    
    def get_train_test_feat_targets(self):
        self.all_ids = list(self.hasG_hide.nodes())
        self.train_ids = self.train_node_index
        self.test_ids = self.test_node_index
        
        self.all_targets_missing=[]
        self.all_targets_feat=[]
        
        self.all_feat = self.hasG_hide.node_features()
        
        for id_i in self.all_ids:
            missing_ids = list(set(self.subG.neighbors(id_i)).difference(list(self.hasG_hide.neighbors(id_i))))
            missing_len = len(missing_ids)
            
            if missing_len > 0:
                if len(missing_ids) <= self.num_pred:
                    zeros = np.zeros((max(0, self.num_pred - missing_len), self.feat_shape))
                    missing_feat_all = np.vstack((self.subG.node_features(missing_ids[:self.num_pred]), zeros)).\
                        reshape((1, self.num_pred, self.feat_shape))
                else:
                    missing_feat_all = np.copy(self.subG.node_features(missing_ids[:self.num_pred])).\
                        reshape((1, self.num_pred, self.feat_shape))        
            else:
                missing_feat_all = np.zeros((1, self.num_pred, self.feat_shape))
            self.all_targets_missing.append(missing_len)
            self.all_targets_feat.append(missing_feat_all)
        self.all_targets_missing = np.asarray(self.all_targets_missing).reshape((-1,1)) 
        self.all_targets_feat = np.asarray(self.all_targets_feat).reshape((-1, self.num_pred, self.feat_shape))
        
        self.edges = np.asarray(self.hasG_hide.edges(use_ilocs=True))
        self.adj = self.get_adj(self.edges,len(self.all_ids))
        self.edges = torch.tensor(self.edges.astype(np.int32))
        self.all_feat=torch.tensor(self.all_feat)
        self.all_targets_missing = torch.tensor(self.all_targets_missing)
        self.all_targets_feat = torch.tensor(self.all_targets_feat)
        
        self.train_ilocs = self.hasG_hide.node_ids_to_ilocs(self.train_ids.tolist()).astype(np.int32)
        self.test_ilocs = self.hasG_hide.node_ids_to_ilocs(self.test_ids.tolist()).astype(np.int32)
        #self.train_ilocs = np.array(self.train_ids, dtype=np.int32)
        #self.test_ilocs = np.array(self.test_ids, dtype=np.int32)
        return self.adj, self.all_feat, self.edges, \
                [self.all_targets_missing, self.all_targets_feat], \
                self.train_ilocs, self.train_ilocs, self.test_ilocs
            
    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    def accuracy_missing(self,output, labels):
        output=output.cpu()
        preds = output.detach().numpy().astype(int)
        correct=0.0
        for pred,label in zip(preds,labels):
            if int(pred)==int(label):
                correct+=1.0
        return correct / len(labels)
    def accuracy(self,pred,true):
        acc=0.0
        for predi,truei in zip(pred,true):
            if torch.argmax(predi) == torch.argmax(truei):
                acc+=1.0
        acc/=len(pred)
        return acc
    
    def l2_feat(self,output,labels):
        output=output.view(-1,self.num_pred,self.feat_shape)
        return F.mse_loss(
            output, labels).float()

    def greedy_l2_feat(self, pred_feats, true_feats, pred_missing, true_missing):
        pred_feats = pred_feats.view(-1, self.num_pred, self.feat_shape)
        return feat_loss.greedy_loss(pred_feats, true_feats, pred_missing, true_missing).unsqueeze(0).mean().float()


    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def train_neighgen(self, epoch):
        self.neighgen.train()
        self.optimizer.zero_grad()
        input_feat = self.all_feat
        input_edge = self.edges
        input_adj = self.adj
        output_missing, output_feat = self.neighgen(input_feat, input_edge, input_adj)
        output_missing = torch.flatten(output_missing)
        output_feat=output_feat.view(len(self.all_ids),self.num_pred,self.feat_shape)
        self.train_ilocs = self.train_ilocs.long()
        loss_train_missing = F.smooth_l1_loss(output_missing[self.train_ilocs].float(),
                                              self.all_targets_missing[self.train_ilocs].reshape(-1).float())
        #print("Complete")
    
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
        '''
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss.item()),
              'missing_train: {:.4f}'.format(acc_train_missing),
              'loss_miss: {:.4f}'.format(loss_train_missing.item()),
              'loss_feat: {:.4f}'.format(loss_train_feat.item()),
              'missing_val: {:.4f}'.format(acc_val_missing),
              'l2_val: {:.4f}'.format(l2_val_feat))
        '''
        
    def train(self):
        for epoch in range(config.gen_epochs):
            self.train_neighgen(epoch)
        
        print("NeighGen Finished!")
        
    def set_fed_model(self):
        self.fed_model = FedLightGCN_Plus(self.neighgen)
    
    def save_fed_model(self):
        torch.save(self.fed_model,self.model_path[0])
        
    def load_fed_model(self):
        if config.cuda:
            self.fed_model = torch.load(self.model_path[0])
        else:
            self.fed_model=torch.load(self.model_path[0],map_location=torch.device('cpu'))


class loc_LightGCNPlus:
    def __init__(self, dataset, testset):
        
        # dataset -> data owner list임

        self.dataset = dataset
        self.testset = testset
        self.top_k = config.top_k
        self.n_owners = config.num_owners
        self.n_samples = config.num_samples
        
        precs = []
        recalls = []
        ndcgs = []
        hits = []
        
        self.adj_mat = []
        
        '''
        for owner_i in range(self.n_owners):
            print('dataset check - ' + str(type(dataset[owner_i])))
            print('adj check - ' + str(dataset[owner_i].adj))
            print('dataset length - ' + str(len(dataset[owner_i].adj)))
            print('datset value length' + str(len(dataset[owner_i].adj.coalesce().values())))
            print('datset indices length' + str(len(dataset[owner_i].adj.coalesce().indices()))) 
            print('dataset edges - ' + str(len(dataset[owner_i].edges)))
        '''

        for owner_i in range(self.n_owners):
            coo = dataset[owner_i].adj.coalesce()
            adj_mat = sp.coo_matrix((np.ones_like(coo.values().cpu().numpy()),
                                     (coo.indices()[0].cpu().numpy(),
                                      coo.indices()[1].cpu().numpy())),
                                    shape=coo.shape).tocsr()
            self.adj_mat.append(adj_mat)     
        
        for owner_i in range(self.n_owners):
            train, valid = train_test_split(self.adj_mat[owner_i], test_size=0.1, random_state=1234)
            lightGCN = LightGCN(train, valid)
            lightGCN.fit()
            
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
            
        print(f"Loc Plus LightGCN: prec@{self.top_k} {avg_prec}, recall@{self.top_k} {avg_recall}, ndcg@{self.top_k} {avg_ndcg}, hits@{self.top_k} {avg_hit}")
        