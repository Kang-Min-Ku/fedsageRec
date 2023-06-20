import sys
sys.path.append('.')
import logging
import networkx as nx
import stellargraph as sg

#temp code to renew fedsage.log
temp_fd = open("/home/wooseok/.conda/envs/fedLightGCN/fedLightGCN-main/fedsage.log","w")
temp_fd.close()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('/home/wooseok/.conda/envs/fedLightGCN/fedLightGCN-main/fedsage.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from src.util import config
logger.info("import config")
from src.util.louvain_networkx import louvain_graph_cut
logger.info("import louvain_networkx")
from src.dataowner import DataOwner
logger.info("import dataowner")
from dataowner.load_data import Data
logger.info("import data")
from src.recommenders import Recommender
logger.info("import recommendation")
from src.loc_lightGCN import loc_LightGCN
logger.info("import local LightGCN")
from src.loc_lightGCNPlus import loc_LightGCNPlus, LocalOwner
logger.info("import local LightGCN PLUS")
from src.fed_lightGCN import fed_LightGCN
logger.info("import fed LightGCN")
from src.fed_lightGCNPlus import fed_LightGCNPlus, train_fedgen
logger.info("info fed LightGCN Plus")
from global_task import Global
logger.info("import global")
from sklearn import preprocessing
logger.info("import preprocessing")


def set_up_system():
    
    if config.dataset == 'amazon-book':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    elif config.dataset == 'gowalla':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    elif config.dataset == 'lastfm':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    elif config.dataset == 'yelp2018':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    elif config.dataset == 'ml-100k':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    elif config.dataset == 'ml-1m':
        dataset = Data(config.load_data, batch_size=config.batch_size)
        test = dataset.convert_to_csr_mat()
        plain_adj, norm_adj, mean_adj,pre_adj = dataset.get_adj_mat()
    
    else:
        print("dataset name does not exist!")
        return

    logger.info("Load Data")
    # Load Data
    
    logger.info("Process Data")
    # Process Data
    
    # Global Task 진행한 후에 Global 결과까지 도출해야함
    global_task = Global(plain_adj, test)
    #print(str(plain_adj))

    # 각 Subgraph로 변환해주고, Neighbor edge 정리
    dataowner_list = []
    local_G, local_nodes_ids = louvain_graph_cut(plain_adj)
    loc_local_G, loc_local_nodes_ids = local_G.copy(), local_nodes_ids.copy()
   
    #print('local nodes ids - ' + str(local_nodes_ids))

    for owner_i in range(config.num_owners):
        #print('local G - ' + str(local_G[owner_i]))
        #print('local nodes - ' + str(len(local_nodes_ids[owner_i])))
        
        do_i = DataOwner(do_id=owner_i, subG=local_G[owner_i], sub_ids=local_nodes_ids[owner_i]) 
        do_i.get_edge_nodes()
        do_i.set_gan_path()
        do_i.save_do_info()
        dataowner_list.append(do_i)
        #print('do - ' + str(dir(do_i.subG)))
    #print('last check')
    #print("Owner i len - " + str(len(dataowner_list)))
    dataowner_list_plus = dataowner_list.copy()
    dataowner_list_fed = dataowner_list.copy()
    
    # Local LGCN -> 각각 Stellargraph에 대해서 Train, Test Set으로 분류하고, Local LightGCN을 진행    
    loc_lightgcn = loc_LightGCN(dataowner_list, test)
    
    # Local LGCN Plus -> 각각 Stellargraph행 대해서 Graph Mending 진행하고, Train, Test Set으로 분류하고 Local LightGCN을 진행
    local_owners = []
    fed_local_owners = []
    
    for owner_i in range(config.num_owners):
        do_i = dataowner_list_plus[owner_i]
        local_gen = LocalOwner(do_id=owner_i, subG=do_i.hasG)
        fed_local_owners.append(local_gen)
        local_gen.train()
        local_owners.append(local_gen)
        #print('local_owners len neighgen - ' + str(len(local_gen.edges)))

    loc_lightgcnplus = loc_LightGCNPlus(local_owners, test)
    
    # fed LGCN -> 각각 Stellargraph에 대해서 Train, Test Set으로 분류하고, fed LightGCN을 진행
    fed_lightgcn = fed_LightGCN(dataowner_list_fed, test, plain_adj) 
    
    # -> Global Model에서 진행하는 건 Local에서 변화가 있을 때만 진행해야할 듯? FedAvg처럼
    # fed LGCN Plus -> 각각 Stellargraph에 대해서 Graph Mending 진행하고, Train, Test Set으로 분류하고 Fed LightGCN을 진행
    
    feat_shape = local_owners[0].feat_shape
    
    for owner in local_owners:
        owner.set_fed_model()
    fed_local_owners = train_fedgen(fed_local_owners, feat_shape)
    
    #for owner_i in range(config.num_owners):
    #    print('Fed Local_owners len neighgen - ' + str(len(fed_local_owners[owner_i].edges)))
    
    fed_lightgcnplus = fed_LightGCNPlus(fed_local_owners, test, plain_adj)

    

set_up_system()