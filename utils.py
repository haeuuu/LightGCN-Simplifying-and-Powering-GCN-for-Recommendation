import torch
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F

def identity_mapping(x):
    return x

activation_map = {
    'relu' : F.relu,
    'leaky' : F.leaky_relu,
    'selu' : F.selu,
    'sigmoid' : F.sigmoid,
    'tanh' : F.tanh,
    'none' : identity_mapping
}

def sparse_to_torch(sp):
    coo = sp.tocoo()
    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    shape = coo.shape

    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def get_degree_inv(graph, edge_types, reverse):
    if reverse:
        degree = sum(graph.in_degrees(etype = str(etype)) for etype in edge_types)
    else:
        degree = sum(graph.out_degrees(etype = str(etype)) for etype in edge_types)

    degree_inv = torch.pow(degree, -0.5)
    degree_inv[torch.isinf(degree_inv)] = 0.
    D_inv = sp.diags(degree_inv.numpy())

    return sparse_to_torch(D_inv)

def get_adjacency(graph, edge_types, ukey = 'user', ikey = 'item'):
    n_users, n_items = graph.num_nodes(ukey), graph.num_nodes(ikey)

    adj_uv = torch.sparse_coo_tensor(size = (n_users, n_items))
    for etype in edge_types:
        adj_uv += graph.adjacency_matrix(etype = str(etype))

    return adj_uv

def degree_noramlization(graph, edge_types, ukey = 'user', ikey = 'item'):
    adj_uv = get_adjacency(graph, edge_types, ukey, ikey)
    degree_u = get_degree_inv(graph, edge_types, False)
    degree_v = get_degree_inv(graph, edge_types, True)

    normed_adj_u = torch.sparse.mm(degree_u, adj_uv)
    normed_adj_u = torch.sparse.mm(normed_adj_u, degree_v)

    normed_adj_v = torch.sparse.mm(degree_v, adj_uv.t())
    normed_adj_v = torch.sparse.mm(normed_adj_v, degree_u)

    return normed_adj_u, normed_adj_v

def add_degree(graph, edge_types, symmetric = True, n_users = None, n_items = None):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))

        return x.unsqueeze(1)

    user_ci = []
    user_cj = []
    item_ci = []
    item_cj = []
    for r in edge_types:
        user_ci.append(graph[f'reverse-{r}'].in_degrees())
        item_ci.append(graph[f'{r}'].in_degrees())
        
        if symmetric:
            user_cj.append(graph[f'{r}'].out_degrees())
            item_cj.append(graph[f'reverse-{r}'].out_degrees())

    user_ci = _calc_norm(sum(user_ci))
    item_ci = _calc_norm(sum(item_ci))

    if symmetric:
        user_cj = _calc_norm(sum(user_cj))
        item_cj = _calc_norm(sum(item_cj))
    else:
        user_cj = torch.ones((n_users,))
        item_cj = torch.ones((n_items,))

    graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
    graph.nodes['item'].data.update({'ci': item_ci, 'cj': item_cj})