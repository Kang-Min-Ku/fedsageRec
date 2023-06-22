import scipy.sparse as sp
import numpy as np

def mean_adj_single(adj):
    # D^-1 * A
    rowsum = np.array(adj.sum(1))

    #d_inv = np.power(rowsum, -1).flatten()
    d_inv = 1/rowsum
    d_inv = d_inv.flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print('generate single-normalized adjacency matrix.')
    return norm_adj.tocoo()

def normalized_adj_single(adj):
    # D^-1/2 * A * D^-1/2
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()