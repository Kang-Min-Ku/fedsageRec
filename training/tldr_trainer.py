import copy
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
import GPUtil
# GPUtil.showUtilization()
from time import time
from data_preprocessing.recommandation.utils import *
from FedML.fedml_core.trainer.model_trainer import ModelTrainer

#for centralized model
class TLDRtrainer(ModelTrainer):
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.train_loss_dict = {}

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        logging.info("------>> :: 2 loss NCL training start :: <<-----")


        ####### case 1 : 50(client) + 40(global) #######
        client_data = train_data
        global_data = args.big_data
        _, _, train_data = self.creat_all_data(client_data, global_data)

        model = self.model
        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_data, valid_data, n_train, n_valid = self.split_valid_data(train_data)

        best_perf = 0.0
        patience = 10
        increase = 10

        for epoch in range(args.epochs):
            t1 = time()
            model.e_step()
            loss_func = self.model.calculate_bpr_only #calculate_bpr_only calculate_loss
            total_loss = 0.
            n_batch = n_train // args.batch_size + 1

            for idx in range(n_batch):
                users, pos_items, neg_items = sample(train_data, args.batch_size, self.model.n_items)
                losses = loss_func(users, pos_items, neg_items)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += float(losses)

            if (epoch % 5) == 1:
                result = self.inner_test(valid_data, train_data, device, args)
                perf = result['recall']
                if perf > best_perf:
                    best_perf = perf
                    best_param = copy.deepcopy(model.state_dict())
                    patience = increase + epoch
            if epoch > patience:
                break

            self.train_loss_dict[epoch] = total_loss
            t2 = time()
            train_loss_output = ''.join(f'epoch = {epoch}, loss = {total_loss}. time = {t2-t1}')
            logging.info(train_loss_output)

        if best_perf != 0.0:
            model.load_state_dict(copy.deepcopy(best_param))

        # model.graph_emb_save()
        # logging.info(" :::: graph emb saved :::")


    def test(self, test_data, train_data, device, args=None):
        # logging.info("----->>>> test <<<<-----")
        t1 = time()
        # Ks = eval(args.Ks)
        model = self.model
        model.to(device)
        # model.graph_embedding_expansion()
        # logging.info("----->>>> graph expension end <<<<-----")
        # print("graph expansion finished")
        model.eval()

        ITEM_NUM = model.n_items
        # self.test_data = test_data
        precision, recall, ndcg = [], [], []

        u_batch_size = args.batch_size * 2
        test_users = list(test_data.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]
            # 아이템 all
            item_batch = range(ITEM_NUM)
            u_g_embeddings, i_g_embeddings, _ = model()
            user_embeddings = u_g_embeddings[user_batch]
            pos_embeddings = i_g_embeddings[item_batch]
            rate_batch = model.rating(user_embeddings, pos_embeddings).detach().cpu()
            # 테스트
            precision_batch, recall_batch, ndcg_batch = self.calc_metrics_at_k(rate_batch, train_data,
                                                                               test_data, user_batch,
                                                                               item_batch)
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

        precision_k = sum(np.concatenate(precision)) / n_test_users
        recall_k = sum(np.concatenate(recall)) / n_test_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_test_users

        result = {'precision': precision_k, 'recall': recall_k, 'ndcg': ndcg_k}
        t2 = time()
        # print(f"resultss = {result}, time = {t2-t1}")

        return result, model

    def calc_metrics_at_k(self, cf_scores, train_user_dict, test_user_dict, user_ids, item_ids):
        K = 100
        test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
        for idx, u in enumerate(user_ids):
            train_pos_item_list = train_user_dict[u]
            test_pos_item_list = test_user_dict[u]
            cf_scores[idx][train_pos_item_list] = 0
            test_pos_item_binary[idx][test_pos_item_list] = 1
        try:
            _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)  # try to speed up the sorting process
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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----->>> test_on_the_server <<<------")
        model_list, score_list = [], []
        pre_list, recall_list, ndcg_list = np.array([]), np.array([]), np.array([])

        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            train_data = train_data_local_dict[client_idx]
            score, model = self.test(test_data, train_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])

            model_list.append(model)
            score_list.append(score)
            logging.info(f'Client {client_idx}, Test score = {score}')

            pre_list = np.append(pre_list, score['precision'])
            recall_list = np.append(recall_list, score['recall'])
            ndcg_list = np.append(ndcg_list, score['ndcg'])

        logging.info(f'::(-ㅁ-)::### final Test ### precision = {np.mean(np.array(pre_list))}, '
                     f'recall = {np.mean(np.array(recall_list))}, ndcg = {np.mean(np.array(ndcg_list))}')

        # with open('local_user_embs.npy', 'wb') as f:
        #     init = 0.0
        #     np.save(f, init, allow_pickle=True)

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        # if models_differ == 0:
        # logging.info('Models match perfectly! :)')

    def split_valid_data(self, train_items):
        train_size = 9
        train_dict, valid_dict = {}, {}
        n_train, n_valid = 0, 0
        for u, i in train_items.items():
            i = np.array(i)
            curr = int(len(i)*train_size/10)
            if curr == 0:
                train_dict[u] = np.array(list(i))
                valid_dict[u] = np.array(list(i))
                n_train += 1
                n_valid += 1
            else:
                train_item = np.random.choice(i, curr, replace=False) #90%
                val_item = np.setdiff1d(i, train_item)

                train_dict[u] = train_item
                valid_dict[u] = val_item

                n_train += len(train_item)
                n_valid += len(val_item)

        print(f'train data : valid data = {n_train}:{n_valid}')


        return train_dict, valid_dict, n_train, n_valid

    def creat_all_data(self, client_data, big_data):

        n_client, n_all = 0, 0
        all_data = {}

        for u, i in client_data.items():
            all_data[u] = i
            n_client += len(i)
            n_all += len(i)

        for u, i in big_data.items():
            if u in all_data:
                all_data[u] = np.concatenate((all_data[u], i))
            else:
                all_data[u] = i
            n_all += len(i)

        # print(f"creat all data size: {n_all}")

        return n_client, n_all, all_data

    def inner_test(self, test_data, train_data, device, args=None):
        # logging.info("----->>>> test <<<<-----")
        t1 = time()
        # Ks = eval(args.Ks)

        model = self.model
        model.to(device)
        model.eval()

        ITEM_NUM = model.n_items
        # self.test_data = test_data
        precision, recall, ndcg = [], [], []

        u_batch_size = args.batch_size * 2
        test_users = list(test_data.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]
            # 아이템 all
            item_batch = range(ITEM_NUM)
            u_g_embeddings, i_g_embeddings, _ = model()
            user_embeddings = u_g_embeddings[user_batch]
            pos_embeddings = i_g_embeddings[item_batch]
            rate_batch = model.rating(user_embeddings, pos_embeddings).detach().cpu()
            # 테스트
            precision_batch, recall_batch, ndcg_batch = self.calc_metrics_at_k(rate_batch, train_data,
                                                                               test_data, user_batch,
                                                                               item_batch)
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

        precision_k = sum(np.concatenate(precision)) / n_test_users
        recall_k = sum(np.concatenate(recall)) / n_test_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_test_users

        result = {'precision': precision_k, 'recall': recall_k, 'ndcg': ndcg_k}
        t2 = time()
        # print(f"inner results = {result}, time = {t2-t1}")

        return result
