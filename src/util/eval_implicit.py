import os
import numpy as np
import pandas as pd
import math
import torch

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from tqdm import tqdm
from IPython import embed
from src.util.metrics import compute_metrics
from src.util import config

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIVLE_DEVICES'] = "0"

def eval_implicit(model, train_data, test_data, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []
    hit_list = []
    
    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
        #for item_id in range(len(train_data.T)):
            train_by_item = train_data[:, item_id]
            #missing_user_ids = np.where(train_by_item == 0)[0]  # missing user_id
            missing_user_ids = np.where(train_by_item.getnnz(axis=1) == 0)[0]
            
            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
        #for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            #missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id
            missing_item_ids = np.where(train_by_user.getnnz(axis=1) == 0)[0]
            
            pred_u_score = pred_matrix[user_id, missing_item_ids]
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k, hit_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
            hit_list.append(hit_k)

    else:
        print('train data - ' + str(train_data.shape))
        
        for user_id in range(len(train_data.shape)):
            train_by_user = train_data[user_id]
            #missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id
            missing_item_ids = np.where(train_by_user.getnnz(axis=0) == 0)[0]

            pred_u_score = model.predict(user_id, missing_item_ids)
            pred_u_idx = np.argsort(pred_u_score)[::-1]  # 내림차순 정렬
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user.getnnz(axis=0) >= 0.5)[0]
            
            prec_k, recall_k, ndcg_k, hit_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
            hit_list.append(hit_k)

    return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list), np.mean(hit_list)



def eval_implicit_NGCF(self, model, trainset, testset, top_k):
    model.eval()
    
    ITEM_NUM = model.n_item
    USER_NUM = model.n_user
    u_batch_size = config.batch_size * 2
    i_batch_size = config.batch_size
    
    #test_users = list(testset.keys())
    test_users = np.unique(testset.tocoo().row)
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    precision, recall, ndcg = [], [], []
    
    for u_batch_id in range(n_user_batchs):
        
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        
        user_batch = np.array(test_users[start: end])
        item_batch = np.array(range(ITEM_NUM))

        user_batch = user_batch[user_batch <= USER_NUM].tolist()
        item_batch = item_batch[item_batch <= ITEM_NUM].tolist()
    
        u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                        item_batch,
                                                        [],
                                                        drop_flag=False)
        rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        
        precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(rate_batch, trainset,
                                                                        testset, user_batch,
                                                                        item_batch)

        precision.append(precision_batch)
        recall.append(recall_batch)
        ndcg.append(ndcg_batch)
    
    precision_k = sum(np.concatenate(precision)) / n_test_users
    recall_k = sum(np.concatenate(recall)) / n_test_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_test_users

    return ndcg_k, precision_k, recall_k


def eval_implicit_NGCF_total(self, model, trainset, testset, top_k):
    model.eval()
    
    ITEM_NUM = model.n_item
    USER_NUM = model.n_user
    u_batch_size = config.batch_size * 2
    i_batch_size = config.batch_size
    
    #test_users = list(testset.keys())
    test_users = np.unique(testset.tocoo().row)
    n_test_users = len(test_users)
    
    n_user_batchs = n_test_users // u_batch_size + 1
    
    precision, recall, ndcg = [], [], []
    
    #for u_batch_id in range(n_user_batchs):
    
    #user_batch = test_users[0: USER_NUM]
    user_batch = np.array(test_users[0: USER_NUM])
    item_batch = np.array(range(ITEM_NUM))
    
    user_batch = user_batch[user_batch <= USER_NUM].tolist()
    item_batch = item_batch[item_batch <= ITEM_NUM].tolist()
    
    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                    item_batch,
                                                    [],
                                                    drop_flag=False)
    rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
    
    precision, recall, ndcg = calc_metrics_at_k(rate_batch, trainset,
                                                testset, user_batch,
                                                item_batch)
    
    #print('precision len - ' + str(len(precision)))
    
    precision = sum(precision) / len(user_batch)
    recall = sum(recall) / len(user_batch)
    ndcg = sum(ndcg) / len(user_batch)

    return ndcg, precision, recall


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids):
    K = 100
    
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = np.unique(train_user_dict[u].tocoo().col)
        test_pos_item_list = np.unique(test_user_dict[u].tocoo().col)
        train_pos_item_list = train_pos_item_list[train_pos_item_list<len(item_ids)]
        test_pos_item_list = test_pos_item_list[test_pos_item_list<len(item_ids)]
        
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1
        
    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit)#, dtype=np.float32)

    precision = precision_at_k_batch(binary_hit, K)
    recall = recall_at_k_batch(binary_hit, K)
    ndcg = ndcg_at_k_batch(binary_hit, K)
    
    return precision, recall, ndcg


def ndcg_at_k_batch(hits, k):
    hits_k = hits[:, :k]
    dcg = np.sum((np.exp2(hits_k) - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((np.exp2(sorted_hits_k) - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    res = (dcg / idcg)
    return res

def precision_at_k_batch(hits, k):
    res = hits[:, :k].mean(axis=1)
    return res

def recall_at_k_batch(hits, k):
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    if np.isnan(res).any():
        return np.zeros_like(res)
    return res
    

def coo_matrix_to_dict(coo_matrix):
    dict_data = {}
    for user_id, item_id in zip(np.unique(coo_matrix.tocoo().row), np.unique(coo_matrix.tocoo().col)):
        if user_id in dict_data:
            dict_data[user_id].append(item_id)
        else:
            dict_data[user_id] = [item_id]
    dict_data = {user_id: np.array(items, dtype=np.int32) for user_id, items in dict_data.items()}
            
    return dict_data