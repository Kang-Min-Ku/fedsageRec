import numpy as np
import torch

def remove_low_data(training_data, dve_model, ori_model):


    user_list = list(training_data.keys())
    item_list = range(ori_model.n_item)
    u_emb, i_emb, _ = ori_model(user_list, item_list, [])
    #전체 임베딩을 넣는것이 맞냐? 레이블... 레이블...pos_emb이냐 i_emb이냐..
    y_hat = ori_model.rating(u_emb, i_emb).detach().cpu()
    y_label = ori_model.norm_adj
    y_hat_input = y_label - y_hat

    dve_score = dve_model(u_emb, i_emb, y_hat_input)
    ans = torch.cat([user_list, dve_score], dim=2)
    #Sorting한 다음...
    return ans



