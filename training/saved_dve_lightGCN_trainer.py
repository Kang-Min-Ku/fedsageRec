import copy

import numpy as np
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')
import GPUtil
# GPUtil.showUtilization()
from time import time
# from model.dvngcf import data_value_evaluator
from model.TLDR import data_value_evaluator
from data_preprocessing.recommandation.utils import *
from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class DVE_lightgcn_Trainer(ModelTrainer):
    def __init__(self, model, args=None):
        self.args = args
        self.final_model = model
        self.pred_model = copy.deepcopy(model)

        self.dve_lr = 0.0007  # 0.0007 # 0.001 0.0001
        self.dve_epoch = 50  # 50
        self.epsilon = 1e-8
        self.best_dve = 0.0
        self.baseline_delta = 0.0
        # self.exploration_threshold = 0.9
        # self.T = 20
        # self.id = 0
        self.ori_model = copy.deepcopy(self.pred_model)  # train으로 학습 val 예측
        self.val_model = copy.deepcopy(self.pred_model)
        # self.oriTrainer = NGCFtrainer(self.ori_model, args.device, "ori model")
        # self.validTrainer = NGCFtrainer(self.val_model, args.device, "valid model")
        self.oriTrainer = TLDRtrainer(self.ori_model, args)
        self.validTrainer = TLDRtrainer(self.val_model, args)

        self.dve_model = data_value_evaluator()

    def get_model_params(self):
        logging.info("get_model_params")
        return self.final_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.final_model.load_state_dict(model_parameters)

    def dve_train(self, client_data, global_data, device, args=None):

    def train(self, client_data, device, args=None):
        logging.info(" ---->>> :: train_start :: <<<----")
        #
        val_data = client_data
        train_data = args.big_data
        test_data = args.inner_test_data

        n_train = args.n_big
        n_valid, n_all, all_data = self.creat_all_data(client_data, train_data)

        # dve train only first round
        if self.best_dve == 0.0:
            logging.info(" ---->>> :: dve_training :: <<<----")
            # 모델들 생성
            # ori_model = copy.deepcopy(self.pred_model) #train으로 학습 val 예측
            # val_model = copy.deepcopy(self.pred_model) #val로 학습 train으로 예측

            # original recall 구하는 모델
            print("ori model training start")
            best_perf = self.oriTrainer.train(train_data, val_data, n_train, 30, 0.007, args)
            valid_performance = best_perf
            print("valid_performance(Aux only) =", valid_performance)

            # Valid hat 구해서 dve 내부에 넣어주는 역할
            print("val model training start")
            self.validTrainer.train(val_data, [], n_valid, 50, 0.008, args)

            '''
            validTrainer.train(all_data, test_data, n_all, 100, 0.004, args)
            val_test_recall = validTrainer.test(all_data, test_data, args)['recall']
            print("client + random data recall", val_test_recall) #0.7249 고정값
            exit(0)
            '''
            self.ori_model.eval()
            self.val_model.eval()

            # dve_model = data_value_evaluator()
            self.dve_model.to(device)
            self.dve_model.train()
            dve_optimizer = optim.Adam(self.dve_model.parameters(), lr=self.dve_lr)

            best_final_model = self.final_model
            best_performance = 0.0
            patience = 50
            increase = 8

            # train data 넣어서 dve 학습
            logging.info("---->>> :: (GDVE) start :: <<<----")
            for epoch in range(self.dve_epoch):
                monitering_loss = 0
                monitering_recall = []

                n_batch = n_train // args.batch_size + 1
                # print("n_print", n_batch) #4971-128 1243-512 621-1024 310-2048
                new_model = copy.deepcopy(self.pred_model)
                # newTrainer = NGCFtrainer(new_model, device, "predict model")
                newTrainer = TLDRtrainer(new_model, args)
                t1 = time()
                for idx in range(n_batch):
                    # with torch.no_grad():
                    users, pos_items, neg_items = sample(train_data, args.batch_size, self.ori_model.n_items)
                    # tldr
                    u_g_embeddings, i_g_embeddings, _ = self.ori_model()
                    u_emb = u_g_embeddings[users]
                    pos_emb = i_g_embeddings[pos_items]

                    y_hat_input = self.validTrainer.predict_score(users)  # : dve만들때 씀 y - y_valid_hat
                    est_dv = self.dve_model(u_emb, pos_emb, y_hat_input.to(device))  # sigmoid 값
                    sel_prob = torch.bernoulli(est_dv).detach()
                    if sel_prob.sum() == 0:
                        est_dv = 0.5 * torch.ones_like(est_dv)
                        sel_prob = torch.bernoulli(est_dv).detach()

                    log_prob = torch.sum(sel_prob * torch.log(est_dv + self.epsilon) +
                                         (1.0 - sel_prob) * torch.log(1.0 - est_dv + self.epsilon))
                    # predmodel dvrl 결과로 학습시키기 pred
                    newTrainer.dvrl_fit(users, pos_items, neg_items, sel_prob, 30, 0.001, loss_flag=1)
                    dvrl_perf = newTrainer.test(val_data, train_data, args)['recall']  # 오래걸림
                    # with torch.no_grad():
                    reward_curr = valid_performance - dvrl_perf  # dvrl perf가 커지는 방향
                    reward_curr = torch.as_tensor(reward_curr).to(device=args.device)
                    extra_threshold = max(torch.mean(est_dv.squeeze()) - 0.9, 0) + max(
                        0.1 - torch.mean(est_dv.squeeze()), 0)

                    dve_optimizer.zero_grad()
                    dve_loss = reward_curr * log_prob + 1e3 * extra_threshold
                    dve_loss.requires_grad_(True)
                    dve_loss.backward()
                    dve_optimizer.step()

                    monitering_recall.append(float(dvrl_perf))
                    monitering_loss += float(dve_loss)

                # early stopping
                perf = np.mean(monitering_recall)
                if perf > best_performance:
                    best_performance = perf
                    # print(f"best recall of dve : {best_performance}")
                    best_param = copy.deepcopy(self.dve_model.state_dict())
                    patience = increase + epoch
                if epoch > patience:
                    break
                t2 = time()
                logging.info(
                    f"epoch = {epoch}, recall: {np.mean(monitering_recall):.5f}, dve_loss={monitering_loss:.4f}, time={t2 - t1:.4f}")

            if best_performance != 0.0:
                self.best_dve = best_performance
                self.dve_model.load_state_dict(copy.deepcopy(best_param))
                # print(f"best dve = {self.best_dve}, hey good job")

            logging.info("---->>> :: (GDVE)finished :: <<<----")

        # final model training with dve
        logging.info("---->>> :: final train start :: <<<----")
        final_epoch = args.epochs  # 10,100
        final_best, patience_ = 0, 10
        # final_Trainer = NGCFtrainer(self.final_model, device, "final model")
        final_Trainer = TLDRtrainer(self.final_model, args)

        self.dve_model.eval()
        sel_data = []
        for epoch in range(final_epoch):
            select_num, ff_loss = 0, 0
            n_batch = n_all // args.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = sample(all_data, args.batch_size, self.ori_model.n_items)  # all_data

                # tldr
                u_embs, i_embs, _ = self.ori_model()
                u_emb = u_embs[users]
                pos_emb = i_embs[pos_items]

                y_hat_input = self.validTrainer.predict_score(users)
                est_dv = self.dve_model(u_emb, pos_emb, y_hat_input.to(device))
                sel_prob = torch.bernoulli(est_dv).detach()
                select_num += sum(sel_prob).cpu()
                f_loss = final_Trainer.dvrl_fit(users, pos_items, neg_items, sel_prob, 1, 0.001, loss_flag=0)  # 0.004
                ff_loss += f_loss
            final_out = final_Trainer.test(test_data, all_data, args)['recall']  # all_data, final_Trainer
            logging.info(f" --->> epoch = {epoch}, recall = {final_out}, loss = {ff_loss}")
            sel_data.append(select_num)

            if final_out > final_best:
                final_best = final_out
                best_final_model = {k: v.cpu() for k, v in
                                    self.final_model.state_dict().items()}  # copy.deepcopy(self.final_model.state_dict())
                patience_ = 8 + epoch
            if epoch > patience_:
                break

        # self.final_model.graph_emb_save()
        # logging.info(" :::: graph emb saved :::")

        # print(f"final best ={final_best:.5f}")
        logging.info(f"select_pers = {np.mean(sel_data):.4f}, {(np.mean(sel_data) / n_all) * 100:.4f}%")
        # print("종료")
        return final_best, best_final_model

    def test(self, test_data, train_data, args=None):
        # logging.info("----->>>> test <<<<-----")

        _, _, all_data = self.creat_all_data(train_data, args.big_data)

        # Ks = eval(args.Ks)
        model = self.final_model
        model.eval()
        # model.graph_embedding_expansion()
        # logging.info("----->>>> graph expension end <<<<-----")
        model.to(args.device)
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

            # tldr
            u_g_embeddings, i_g_embeddings, _ = model()
            user_embeddings = u_g_embeddings[user_batch]
            pos_embeddings = i_g_embeddings[item_batch]
            rate_batch = model.rating(user_embeddings, pos_embeddings).detach().cpu()

            precision_batch, recall_batch, ndcg_batch = self.calc_metrics_at_k(rate_batch, all_data,
                                                                               test_data, user_batch,
                                                                               item_batch)

            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

        precision_k = sum(np.concatenate(precision)) / n_test_users
        recall_k = sum(np.concatenate(recall)) / n_test_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_test_users

        result = {'precision': precision_k, 'recall': recall_k, 'ndcg': ndcg_k}
        # print("resultss =", result)
        return result, model

    def calc_metrics_at_k(self, cf_scores, train_user_dict, test_user_dict, user_ids, item_ids):
        K = 100
        test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
        for idx, u in enumerate(user_ids):
            # print(u)
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
            score, model = self.test(test_data, train_data, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])

            model_list.append(model)
            score_list.append(score)
            logging.info(f'Client {client_idx}, Test score = {score}')

            pre_list = np.append(pre_list, score['precision'])
            recall_list = np.append(recall_list, score['recall'])
            ndcg_list = np.append(ndcg_list, score['ndcg'])

        logging.info(f'*(^a^)*### final Test ### precision Score = {np.mean(np.array(pre_list))}, '
                     f'recall Score = {np.mean(np.array(recall_list))}, ndcg Score = {np.mean(np.array(ndcg_list))}')
        # .format(np.average(pre_list, axis=0), np.average(recall_list, axis=0), np.average(ndcg_list, axis=0)))
        # wandb.log({"Test/recall": avg_score})

        # with open('local_user_embs.npy', 'wb') as f:
        #     logging.info('init local embs to 0.0')
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


###############################################

# for fed condition
class TLDRtrainer():
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.device = args.device
        self.train_loss_dict = {}

    def train(self, train_data, valid_data, n_train, epochs, train_lr, args=None):
        logging.info("------>> :: 2 loss NCL training start :: <<-----")

        model = self.model
        model.to(args.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=train_lr)

        best_perf = 0.0
        patience = 10
        increase = 8

        for epoch in range(epochs):
            t1 = time()

            loss_func = model.calculate_loss
            total_loss = 0.

            n_batch = n_train // args.batch_size + 1

            for idx in range(n_batch):
                users, pos_items, neg_items = sample(train_data, args.batch_size, model.n_items)
                losses = loss_func(users, pos_items, neg_items)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += float(losses)

            if (epoch % 2) == 0:
                # logging.info("...모델 학습 중 epoch={} loss= {}".format(epoch, loss))
                if valid_data != []:
                    result = self.test(valid_data, train_data, args)
                    perf = result['recall']
                    if perf > best_perf:
                        best_perf = perf
                        best_param = copy.deepcopy(model.state_dict())
                        patience = increase + epoch
            if valid_data != [] and epoch > patience:
                break

            if best_perf != 0.0:
                model.load_state_dict(copy.deepcopy(best_param))

            self.train_loss_dict[epoch] = total_loss
            t2 = time()
            train_loss_output = ''.join(f'epoch = {epoch}, loss = {total_loss}. time = {t2 - t1}')
            logging.info(train_loss_output)
        return best_perf

    def test(self, test_data, train_data, args=None):
        # logging.info("----->>>> test <<<<-----")
        t1 = time()
        # Ks = eval(args.Ks)
        model = self.model
        model.to(self.device)
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

        return result

    def calc_metrics_at_k(self, cf_scores, train_user_dict, test_user_dict, user_ids, item_ids):
        K = 100
        test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
        for idx, u in enumerate(user_ids):
            # print(u)
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

    def predict_score(self, users, args=None):

        model = self.model
        model.to(self.device)
        n_item = model.n_items

        u_emb, i_emb, _ = model()
        u_emb = u_emb[users]
        y_hat = model.rating(u_emb, i_emb).detach().cpu()  # matmul u,i
        y_label = model.norm_adj[users, model.n_users:]  # 이게 맞을까
        y_label = torch.from_numpy(y_label.todense())

        return np.abs(y_label - y_hat).float()

    def dvrl_fit(self, users, pos_items, neg_items, sel_prob, inner_epochs, dve_lr, loss_flag):

        model = self.model
        model.to(self.device)
        model.train()

        sel_prob = sel_prob.to(torch.device('cpu'))
        sel_prob = sel_prob.tolist()

        user, pos_item, neg_item = [], [], []
        for i in range(len(users)):
            if sel_prob[i] == 1.0:
                user.append(users[i])
                pos_item.append(pos_items[i])
                neg_item.append(neg_items[i])

        best_loss = float('inf')
        patience = 15
        increase = 8
        optimizer = optim.Adam(model.parameters(), lr=dve_lr)  # 0.005

        loss_func = model.calculate_loss

        for epoch in range(inner_epochs):
            losses = loss_func(users, pos_items, neg_items)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if losses < best_loss:
                best_loss = losses
                best_param = copy.deepcopy(model.state_dict())
                patience = increase + epoch
            if epoch > patience:
                break
            # logging.info(f"predict 40번 학습 loss ={float(loss):.4f}")
        # model.load_state_dict(copy.deepcopy(best_param))

        return float(losses)