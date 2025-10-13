import logging
import numpy as np
import torch
import torch.optim as optim
from data_preprocessing.recommandation.utils import *
import warnings
warnings.filterwarnings('ignore')
from time import time
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
# import torch.multiprocessing as mp
#
# cores = mp.cpu_count() // 2

class NGCFTrainer(ModelTrainer):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def get_model_params(self):
        return self.model.state_dict() #self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        model = self.model
        model.to(device)
        model.train()
        logging.info(" ---->>> :: train_start :: <<<----")
        n_train_data = args.data_generator.n_train
        self.training_data = train_data

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        cur_best_pre_0 = 0
        best_param = {}

        for epoch in range(args.epochs):
            t1 = time()
            loss, mf_loss, emb_loss = 0., 0., 0.
            n_batch = n_train_data // args.batch_size + 1

            for idx in range(n_batch):
                users, pos_items, neg_items = args.data_generator.sample(train_data, args)
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                               pos_items,
                                                                               neg_items,
                                                                               drop_flag=args.node_dropout_flag) #forward

                batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                  pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            t2 = time()

            logging.info("epoch : {}, loss : {}, time : {}, n_batch : {}".format(epoch, loss, t2 - t1, n_batch))

        return cur_best_pre_0, best_param


    def calc_metrics_at_k(self, cf_scores, train_user_dict, test_user_dict, user_ids, item_ids):
        K = 100
        test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
        for idx, u in enumerate(user_ids):
            #print(u)
            train_pos_item_list = train_user_dict[u]
            test_pos_item_list = test_user_dict[u]
            cf_scores[idx][train_pos_item_list] = 0
            test_pos_item_binary[idx][test_pos_item_list] = 1
        try:
            _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
        except:
            _, rank_indices = torch.sort(cf_scores, descending=True)
        rank_indices = rank_indices.cpu()

        binary_hit = []
        for i in range(len(user_ids)):
            binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
        binary_hit = np.array(binary_hit, dtype=np.float32)

        precision = precision_at_k_batch(binary_hit, K)
        recall = recall_at_k_batch(binary_hit, K)
        ndcg = ndcg_at_k_batch(binary_hit, K)
        return precision, recall, ndcg


    def test(self, test_data, train_data, device, args=None, drop_flag=False):
        logging.info("----->>>> test <<<<-----")

        Ks = eval(args.Ks)
        model = self.model
        model.to(device)
        ITEM_NUM = model.n_item

        # self.test_data = test_data

        # result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
        #           'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

        u_batch_size = args.batch_size * 2
        i_batch_size = args.batch_size

        test_users = list(test_data.keys())
        n_test_users = len(test_users)

        n_user_batchs = n_test_users // u_batch_size + 1

        precision, recall, ndcg = [], [], []

        for u_batch_id in range(n_user_batchs):

            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]

            # 아이템배치
            # n_item_batchs = ITEM_NUM // i_batch_size + 1
            # rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
            # i_count = 0
            # for i_batch_id in range(n_item_batchs):
            #     i_start = i_batch_id * i_batch_size
            #     i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
            #
            #     item_batch = range(i_start, i_end)
            #
            #     if drop_flag == False:
            #         u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
            #                                                       item_batch,
            #                                                       [],
            #                                                       drop_flag=False)
            #         i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            #     else:
            #         u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
            #                                                       item_batch,
            #                                                       [],
            #                                                       drop_flag=True)
            #         i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            #
            #     rate_batch[:, i_start: i_end] = i_rate_batch
            #     i_count += i_rate_batch.shape[1]
            # assert i_count == ITEM_NUM

            #아이템 all
            item_batch = range(ITEM_NUM)

            u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                          item_batch,
                                                          [],
                                                          drop_flag=False)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

            #테스트
            # user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
            # batch_result = map(self.test_one_user, user_batch_rating_uid)

            precision_batch, recall_batch, ndcg_batch = self.calc_metrics_at_k(rate_batch, train_data,
                                                                               test_data, user_batch,
                                                                               item_batch)

            # batch_results = list(batch_result) #2048개
            # count += len(batch_results)

            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

            # for re in batch_results:
            #     result['precision'] += re['precision'] / n_test_users
            #     result['recall'] += re['recall'] / n_test_users
            #     result['ndcg'] += re['ndcg'] / n_test_users
            #     result['hit_ratio'] += re['hit_ratio'] / n_test_users
            #     result['auc'] += re['auc'] / n_test_users

        # assert count == n_test_users
        # self.result = result
        # pool.close()
        precision_k = sum(np.concatenate(precision)) / n_test_users
        recall_k = sum(np.concatenate(recall)) / n_test_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_test_users

        result = {'precision': precision_k, 'recall': recall_k, 'ndcg': ndcg_k}
        print(f'result = {result}')
        return result, model #score, model


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----->>> test_on_the_server <<<------")
        model_list, score_list = [], []
        pre_list, recall_list, ndcg_list, hit_list, auc_list = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            train_data = train_data_local_dict[client_idx]
            score, model = self.test(test_data, train_data, device, args)

            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])

            model_list.append(model)
            score_list.append(score)
            logging.info('Client {}, Test score = {}'.format(client_idx, score))

            pre_list = np.append(pre_list, score['precision'])
            recall_list = np.append(recall_list, score['recall'])
            ndcg_list = np.append(ndcg_list, score['ndcg'])
            # hit_list = np.append(hit_list, score['hit_ratio'])
            # auc_list = np.append(auc_list, score['auc'])
            #wandb.log({"Client {} Test/recall".format(client_idx): score})

        logging.info('*(^a^)*### final Test ### precision Score = {}, recall Score = {}, ndcg Score = {}'.format(
            np.average(pre_list, axis=0), np.average(recall_list, axis=0), np.average(ndcg_list, axis=0)))
        #wandb.log({"Test/recall": avg_score})
        return True


    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        # if models_differ == 0:
            # logging.info('Models match perfectly! :)')
