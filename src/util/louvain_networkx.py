from community import community_louvain
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
import scipy.sparse as sp
import stellargraph as sg
from src.util import config
import matplotlib.pyplot as plt
from dataowner.load_data import Data
from networkx.algorithms.community import greedy_modularity_communities

def louvain_graph_cut(adj_mat):
        
    # Networkx graph
    adj_G = nx.from_scipy_sparse_matrix(adj_mat)
    # Stellargraph graph
    sG = sg.StellarGraph.from_networkx(adj_G)
    G = sg.StellarGraph.to_networkx(sG)
    
    edges = np.copy(sG.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]
    
    partition = community_louvain.best_partition(G)
    # partition = greedy_modularity_communities(G, n_communities=config.num_owners)
    
    groups = []
    
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    #print(groups)
    partition_groups = {group_i:[] for group_i in groups}
    
    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    
    group_len_max = len(list(sG.nodes()))//config.num_owners-config.delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    
    #print(groups)
    
    len_list=[]
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
        
    len_dict={}
    
    for i in range(len(groups)):
        len_dict[groups[i]]=len_list[i]
    sort_len_dict={k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1],reverse=True)}

    owner_node_ids={owner_id:[] for owner_id in range(config.num_owners)}

    owner_nodes_len=len(list(G.nodes())) //config.num_owners
    owner_list=[i for i in range(config.num_owners)]
    owner_ind=0
    ind_check=0
 
    for group_i in sort_len_dict.keys():
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
        
        if (ind_check == 0):
            owner_ind +=1
            if (owner_ind == config.num_owners):
                ind_check = 1
        
        if (ind_check == 1):
            owner_ind -=1
            if (owner_ind == 0):
                ind_check = 0
    
    for owner_i in owner_node_ids.keys():
        print('nodes len for '+str(owner_i)+' = '+str(len(owner_node_ids[owner_i])))

    nodes_id = sG.nodes()
    local_G = []
    local_nodes_ids = []
    
    # Node Degree를 Feature로 갖는 Local Graph
    for owner_i in range(config.num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = sG.node_ids_to_ilocs(partition_i)
        local_nodes_ids.append(partition_i)

        degree = []
        for node in locs_i:
            degree.append(nx.classes.function.degree(G, node))
            
        feats_i = np.zeros(sG.node_features().shape[0])    
        degree_list = np.array(degree)  
        
        for i, node in enumerate(locs_i):
            feats_i[node] = degree_list[i]
        
        nodes = sg.IndexedArray(feats_i.reshape(-1, 1), nodes_id)
        graph_i = sg.StellarGraph(nodes=nodes, edges=df)
        local_G.append(graph_i)
        
        #print('node features - ' + str(graph_i.node_features()))
        #print('degree - ' + str(degree))
        #print('degree len - ' + str(len(degree)))
        #print('locs_i len - ' + str(len(locs_i)))
    
    return local_G, local_nodes_ids