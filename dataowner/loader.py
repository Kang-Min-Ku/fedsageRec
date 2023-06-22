import logging
import os
import numpy as np
import random as rd

import scipy.sparse as sp
from scipy.stats import beta, gamma, poisson
from time import time

import logging
import os
import numpy as np
import pandas as pd
import random as rd
import scipy.sparse as sp
from scipy.stats import beta, gamma, poisson
from time import time
import stellargraph as sg
import networkx as nx

class DataGenerator(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        #train 갯수
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        #test 갯수
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) #dictionary of keys matrix

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items #train set 딕셔너리

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
    
    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def split_bigdata(self, train_items, big_size=5):
        #big_size = 5
        big_dict, train_dict, inner_test_dict = {}, {}, {}
        n_big, n_train, n_test = 0, 0, 0
        for u, i in train_items.items():
            i = np.array(i)
            curr = int(len(i)*big_size/10)
            if curr == 0:
                big_dict[u] = np.array(list(i))
                train_dict[u] = np.array(list(i))
                n_big += 1
                n_train += 1
            else:
                big_item = np.random.choice(i, curr, replace=False)
                rest_item = np.setdiff1d(i, big_item)
                # test_item = np.random.choice(rest_item, len(rest_item) // (10 - big_size), replace=False) #10%
                # train_item = np.setdiff1d(rest_item, test_item)

                big_dict[u] = big_item
                # inner_test_dict[u] = test_item
                train_dict[u] = rest_item #train_item

                n_big += len(big_item)
                n_train += len(rest_item) #len(train_item)
                # n_test += len(test_item)

        # print(f">>- - - @@@ Auxiliary big data size ={n_big}, total_client data = {n_train} - - - <<")
        self.big_data = big_dict
        # self.inner_test_data = inner_test_dict
        self.n_big = n_big
        # self.n_inner_test = n_test

        return n_train, train_dict
    
    def global_local_partition(self, client_number, combine_global=True, global_data_ratio=0.5):
        num_users, num_items = self.n_users, self.n_items
        num_nodes = num_users + num_items
        train_items = self.train_items
        test_items = {k:np.array(v) for k, v in self.test_set.items()}

        global_data_ratio *= 10
        new_train, global_train_items = self.split_bigdata(train_items)
        local_train_items = {}

        def list_subtraction(src, target):
            return list(set(src) - set(target))

        local_train_items = {k: np.array(list_subtraction(v, global_train_items[k])) for k, v in train_items.items()}
        new_idx = list(range(num_users))
        rd.shuffle(new_idx)

        client_train_dicts = [None] * client_number
        client_indices = np.array_split(new_idx, client_number)

        for c in range(client_number):
            client_train_datasets = {}
            client_valid_datsets = {}
            client_test_datasets = {}

            for user in range(num_users):
                if not combine_global and user in client_indices[c]:
                    client_train_datasets[user] = local_train_items[user]
                    continue

                if user in client_indices[c]:
                    client_train_datasets[user] = np.concatenate([global_train_items[user], local_train_items[user]])
                    try:
                        client_test_datasets[user] = test_items[user]
                    except Exception:
                        continue
                else:
                    client_train_datasets[user] = global_train_items[user]

            partition = {'train': client_train_datasets,
                         'valid': client_valid_datsets,
                         'test': client_test_datasets
                        }
            
            client_train_dicts[c] = partition

        return client_train_dicts, client_indices, local_train_items, global_train_items, test_items
    
    def convert_user_item_to_node_view(self, R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        #print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        # t2 = time()
        # def normalized_adj_single(adj):
        #     rowsum = np.array(adj.sum(1))

        #     d_inv = np.power(rowsum, -1).flatten()
        #     d_inv[np.isinf(d_inv)] = 0.
        #     d_mat_inv = sp.diags(d_inv)

        #     norm_adj = d_mat_inv.dot(adj)
        #     #print('generate single-normalized adjacency matrix.')
        #     return norm_adj.tocoo()

        # def check_adj_if_equal(adj):
        #     dense_A = np.array(adj.todense())
        #     degree = np.sum(dense_A, axis=1, keepdims=False)

        #     temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        #     #print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        #     return temp
        
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # mean_adj_mat = normalized_adj_single(adj_mat)
        
        #print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr() # , norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
    
    def convert_dict_to_csr_mat(self, dict):
        row_ind = [k for k, v in dict.items() for _ in range(len(v))]
        col_ind = [i for ids in dict.values() for i in ids]
        csr_mat = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)), shape=(self.n_users, self.n_items))
        
        return csr_mat    

    def generate(self, client_number, combine=True, global_data_ratio=0.5):
        client_train_dicts, client_indices, _, _, test_items = self.global_local_partition(client_number, combine, global_data_ratio)
        local_G = []
        local_nodes_ids = []
        for c in range(client_number):
            c_R = self.convert_dict_to_csr_mat(client_train_dicts[c]["train"])
            c_A = self.convert_user_item_to_node_view(c_R)
            adj_G = nx.from_scipy_sparse_matrix(c_A)
            
            nodes_ids = list(client_indices[c]) + list(range(self.n_users, self.n_users+self.n_items))

            sG = sg.StellarGraph.from_networkx(adj_G)
            ### add degree feature
            G = sg.StellarGraph.to_networkx(sG)
            degree = []
            for node in nodes_ids:
                degree.append(nx.classes.function.degree(G, node))

            feats = np.zeros(sG.node_features().shape[0])
            degree_list = np.array(degree)

            for i, node in enumerate(nodes_ids):
                feats[node] = degree_list[i]

            nodes = sg.IndexedArray(feats.reshape(-1,1), index=sG.nodes())
            edges = pd.DataFrame(sG.edges(), columns=["source", "target"])
            sG = sg.StellarGraph(nodes=nodes, edges=edges)
            ###
            local_G.append(sG)
            local_nodes_ids.append(nodes_ids)

        test = self.convert_dict_to_csr_mat(test_items)

        return local_G, local_nodes_ids, test

#fedsage loader
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        
                    self.train_items[uid] = train_items
                    
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)
                
        return split_uids, split_state
    
    
    def convert_to_csr_mat(self):
        
        row_ind = [k for k, v in self.test_set.items() for _ in range(len(v))]
        col_ind = [i for ids in self.test_set.values() for i in ids]
        csr_mat = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))
        
        return csr_mat    