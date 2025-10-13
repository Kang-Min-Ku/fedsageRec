import numpy as np
from sklearn.metrics import roc_auc_score
import heapq
import logging
import random as rd
from collections import defaultdict


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def get_performance(user_pos_test, r, auc, KKs):
    metric = metrics()
    # precision, recall, ndcg, hit_ratio = [], [], [], []
    precision, recall, ndcg = [], [], []
    for K in KKs:
        precision.append(metric.precision_at_k(r,K))
        recall.append(metric.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metric.ndcg_at_k(r, K, user_pos_test))
        # hit_ratio.append(metric.hit_at_k(r,K))
    # {'recall': np.array(recall), 'precision': np.array(precision),
    #  'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}
    return {'precision': np.array(precision), 'recall': np.array(recall), 'ndcg': np.array(ndcg)}

def get_auc(item_score, user_pos_test):
    metric = metrics()
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metric.AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    #안씀
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score: #y_hat pos item list
        if i in user_pos_test: # real pos 아이템 리스트
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def ndcg_at_k_batch(hits, k):
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res

def precision_at_k_batch(hits, k):
    res = hits[:, :k].mean(axis=1)
    return res

def recall_at_k_batch(hits, k):
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res

def ranklist_by_heapq_r(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

class metrics(object):

    def recall(self, rank, ground_truth, N):
        return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k]
        return np.mean(r)

    def average_precision(self, r, cut):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        Returns:
            Average precision
        """
        r = np.asarray(r)
        out = [self.precision_at_k(r, k + 1) for k in range(cut) if r[k]]
        if not out:
            return 0.
        return np.sum(out) / float(min(cut, np.sum(r)))

    def mean_average_precision(self, rs):
        """Score is mean average precision
        Relevance is binary (nonzero is relevant).
        Returns:
            Mean average precision
        """
        return np.mean([self.average_precision(r) for r in rs])

    def dcg_at_k(self, r, k, method=1):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    def ndcg_at_k(self, r, k, ground_truth, method=1):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Normalized discounted cumulative gain

            Low but correct defination
        """
        GT = set(ground_truth)
        if len(GT) > k:
            sent_list = [1.0] * k
        else:
            sent_list = [1.0] * len(GT) + [0.0] * (k - len(GT))
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method) #self.dcg_at_k(sent_list, k, method)
        if not dcg_max:
            return 0.
        return (self.dcg_at_k(r, k, method) / dcg_max)

    def recall_at_k(self, r, k, all_pos_num):
        # if all_pos_num == 0:
        #     return 0
        r = np.asfarray(r)[:k]
        return (np.sum(r) / all_pos_num)

    def hit_at_k(self, r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def F1(self, pre, rec):
        if pre + rec > 0:
            return (2.0 * pre * rec) / (pre + rec)
        else:
            return 0.

    def AUC(self, ground_truth, prediction):
        try:
            res = roc_auc_score(y_true=ground_truth, y_score=prediction)
        except Exception:
            res = 0.
        return res

def sample(train_data, batch_size, n_items):

    if batch_size <= len(train_data):
        users = rd.sample(train_data.keys(), batch_size)  # 클라이언트 유저중에서 배치 사이즈만큼만 뽑음
    else:
        users = [rd.choice(list(train_data.keys())) for _ in range(batch_size)]  # 배치 사이즈가 유저 수 보다 크면 중복해서 랜덤하게 뽑으새욤

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
            neg_id = np.random.randint(low=0, high=n_items, size=1)[0]  # 아이템중 아무거나 랜덤하게 하나 고름
            if neg_id not in train_data[u] and neg_id not in neg_items:  # train item에 들어있으면 pos이고 안들어있으면 neg임.
                neg_items.append(neg_id)
        return neg_items

    pos_items, neg_items = [], []
    for u in users:
        pos_items += sample_pos_items_for_u(u, 1)
        neg_items += sample_neg_items_for_u(u, 1)

    return users, pos_items, neg_items


