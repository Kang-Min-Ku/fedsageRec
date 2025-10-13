import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import beta, gamma, poisson
import matplotlib.pyplot as plt


def extract_graph_info(graph: nx.Graph, weights: np.ndarray = None) -> Tuple[np.ndarray, csr_matrix, Dict]:
    """
    Plot adjacency matrix of a graph as a pdf file
    Args:
        graph: the graph instance generated via networkx
        weights: the weights of edge
    Returns:
        probs: the distribution of nodes
        adj: adjacency matrix
        idx2node: a dictionary {key: idx, value: node name}
    """
    idx2node = {}
    for i in range(len(graph.nodes)):
        idx2node[i] = i

    probs = np.zeros((len(graph.nodes), 1))
    adj = lil_matrix((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        src = edge[0]
        dst = edge[1]
        if weights is None:
            adj[src, dst] += 1
            probs[src, 0] += 1
            probs[dst, 0] += 1
        else:
            adj[src, dst] += weights[src, dst]
            probs[src, 0] += weights[src, dst]
            probs[dst, 0] += weights[src, dst]

    return probs, csr_matrix(adj), idx2node


# num_nodes = 100
#
# G = nx.gaussian_random_partition_graph(n=num_nodes, s=4, v=5, p_in=0.25, p_out=0.1)
# p_s, cost_s, idx2node = extract_graph_info(G)
# #p_s = Distribution of nodes in a graph
# list1 = np.array(p_s).flatten().tolist()
# print(f'graph probabiltiy: {list1}')
#
# # p_s = (p_s + 1) ** 0.01
# # p_s /= np.sum(p_s)
# gt = np.zeros((num_nodes,))
# for i in range(len(G.nodes)):
#     gt[i] = G.nodes[i]['block']
# num_partitions = int(np.max(gt)+1)
# # print(gt)
#
# color_chart = {0: "black", 1: "gray", 2: "silver", 3: "cadetblue", 4: "lightslategray", 5: 'goldenrod'}
# color_map = []
#
# # for i in range(len(gt)):
# #     color_map.append(color_chart[gt[i]])
# # print(color_map)
#
# # nx.draw_networkx(G, nx.spring_layout(G))
# # nx.draw(G, pos=nx.spring_layout(G), node_color = color_map)
# # plt.tight_layout()
#
# mean = sum(list1)/len(list1)
# var = np.var(list1)
# plt.hist(list1)
# plt.show()
#
# x = np.arange(8, dtype='f8')
# b = var / mean
# a = mean / b
# print(f'a,b = {a,b}')
# y = gamma.pdf(x, a)
# plt.plot(x, y)
# plt.title("gamma")
# plt.show()

import random

l = [1,1,1,1,1,2,2,2,3,3,4,5]
y = np.random.dirichlet(l)
print(y)
random.shuffle(l, lambda: y[0])
print(l)