import logging
import os
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

import matplotlib.pyplot as plt
import torch.utils.data as data
from data_preprocessing.recommandation.datasets import ngcfdataset
from FedML.fedml_core.non_iid_partition.noniid_partition import partition_class_samples_with_dirichlet_distribution

class Data_generator(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/new_train.txt'
        test_file = path + '/new_test.txt'

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

    def get_adj_mat(self):
        try: #만들어놓은 adj가 있으면 쓰고 없으면 마세요
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + '/adj_mat_full.npz')
            print('already load adj matrix', norm_adj_mat.shape, time() - t1)

        except Exception:
            norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/adj_mat_full.npz', norm_adj_mat)

        return norm_adj_mat


    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def valid_split(self):
        train_dict = {}
        valid_dict = {}
        n_valid = 0

        n_data = self.n_train

        for k,v in self.train_items.items():
            v = np.array(v)
            number = int(0.9*len(v))
            if number == 0:
                # print("split problem...only 1 items!"+str(len(v)))
                train_dict[k] = np.array(list(v))
                valid_dict[k] = np.array(list(v))
                n_valid += 1
            else:
                train_item = np.random.choice(v, number, replace=False)
                valid_item = np.setdiff1d(v, train_item)
                train_dict[k] = train_item
                valid_dict[k] = valid_item
                n_valid += len(valid_item)

        n_train = n_data - n_valid

        return n_train, n_valid, train_dict, valid_dict

    def create_non_uniform_split(self, idxs, client_number, is_train=True):
        logging.info("create_non_uniform_split------------------------------------------")
        N = len(idxs)
        alpha = 0.5
        logging.info("sample number = %d, client_number = %d" % (N, client_number))
        # logging.info(idxs)
        idx_batch_per_client = [[] for _ in range(client_number)]
        idx_batch_per_client, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_number,
                                                                                             idx_batch_per_client, idxs)
        # logging.info(idx_batch_per_client)
        sample_num_distribution = []

        for client_id in range(client_number):
            sample_num_distribution.append(len(idx_batch_per_client[client_id]))
            # logging.info("client_id = %d, sample_number = %d" % (client_id, len(idx_batch_per_client[client_id])))
        logging.info("create_non_uniform_split******************************************")

        # plot the (#client, #sample) distribution
        if is_train:
            logging.info(sample_num_distribution)
            plt.hist(sample_num_distribution)
            plt.title("Sample Number Distribution")
            plt.xlabel('number of samples')
            plt.ylabel("number of clients")
            fig_name = "x_hist.png"
            fig_dir = os.path.join("./visualization", fig_name)
            plt.savefig(fig_dir)
        return idx_batch_per_client


    def partition_data(self, client_number, uniform=True):

        new_train = self.n_train
        n_valid = 0
        train_items = self.train_items
        valid_items = {}
        test_items = self.test_set

        new_idx = list(range(self.n_users))
        rd.shuffle(new_idx)

        client_data_dicts = [None] * client_number

        if uniform:
            clients_idxs = np.array_split(new_idx, client_number)
        else:
            clients_idxs = self.create_non_uniform_split(new_idx, client_number, True)

        for client in range(client_number):
            client_train_datasets = {}
            client_test_datasets = {}
            client_valid_datasets = {}
            for user in clients_idxs[client]:
                client_train_datasets[user] = train_items[user]
                try:
                    client_test_datasets[user] = test_items[user]
                except Exception:
                    continue

            partition_dict = {'train': client_train_datasets,
                              'test': client_test_datasets,
                              'val': client_valid_datasets}

            client_data_dicts[client] = partition_dict

        global_data_dict = {'train': train_items,
                            'test': test_items,
                            'val': valid_items}

        #고쳣슴
        logging.info("----->>> split data set created <<<------")

        return new_train, n_valid, global_data_dict, client_data_dicts



    def load_partition_data(self, client_number, uniform=True):

        data_local_num_dict = {}
        train_data_local_dict = {}
        val_data_local_dict = {}
        test_data_local_dict = {}

        new_train, n_valid, global_data_dict, client_data_dicts = self.partition_data(client_number, uniform)

        train_data_num = new_train
        val_data_num = n_valid
        test_data_num = self.n_test

        train_data_global = global_data_dict['train']
        val_data_global = global_data_dict['val']
        test_data_global = global_data_dict['test']

        for client in range(client_number):
            train_dataset_client = client_data_dicts[client]['train']
            val_dataset_client = client_data_dicts[client]['val']
            test_dataset_client = client_data_dicts[client]['test']

            data_local_num_dict[client] = len(train_dataset_client)

            train_data_local_dict[client] = train_dataset_client
            val_data_local_dict[client] = val_dataset_client
            test_data_local_dict[client] = test_dataset_client

            logging.info("Client idx = {}, local sample number = {}".format(client, len(train_dataset_client)))

        return train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global, \
               data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict


    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def sample(self, train_data, args):

        if args.batch_size <= len(train_data):
            users = rd.sample(train_data.keys(), args.batch_size)  # 클라이언트 유저중에서 배치 사이즈만큼만 뽑음
        else:
            users = [rd.choice(list(train_data.keys())) for _ in range(args.batch_size)]  # 배치 사이즈가 유저 수 보다 크면 중복해서 랜덤하게 뽑으새욤

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = train_data[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]  # 딕셔너리 아니고 인덱스

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:  # num=1
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]  # 아이템중 아무거나 랜덤하게 하나 고름
                if neg_id not in train_data[u] and neg_id not in neg_items:  # train item에 들어있으면 pos이고 안들어있으면 neg임.
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u])) #TODO
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items