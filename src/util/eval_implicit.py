import os
import numpy as np
import pandas as pd
import math

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from IPython import embed
from src.util.metrics import compute_metrics

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
