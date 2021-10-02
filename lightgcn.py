import torch
from torch import nn

import dgl
from dgl import function as fn


class LightGraphConv(nn.Module):
    def __init__(self,
                drop_out = 0.1):
        """Light Graph Convolution

        Paramters
        ---------
        drop_out : float
            dropout rate (neighborhood dropout)
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, graph, feats):
        """Apply Light Graph Convoluiton to specific edge type {r}

        Paramters
        ---------
        graph : dgl.graph
        src_feats : torch.FloatTensor
            source node features

        ci : torch.LongTensor
            in-degree of sources ** (-1/2)
            shape : (n_sources, 1)
        cj : torch.LongTensor
            out-degree of destinations ** (-1/2)
            shape : (n_destinations, 1)

        Returns
        -------
        output : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
                where N_{i, r} ; number of neighbors_{i, r} ** (1/2)
        2. aggregation
            \sum_{j \in N(i), r} MP_{j -> i, r}
        """
        if isinstance(feats, tuple):
            src_feats, dst_feats = feats

        with graph.local_scope():
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']

            cj_dropout = self.dropout(cj)
            weighted_feats = torch.mul(src_feats, cj_dropout)
            graph.srcdata['h'] = weighted_feats

            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'out'))
            out = torch.mul(graph.dstdata['out'], ci)
            
        return out

class LightGCNLayer(nn.Module):
    def __init__(self,
                edge_types,
                drop_out = 0.1):
        super().__init__()
        """LightGCN Layer

        edge_types : list
            all edge types
        drop_out : float
            dropout rate (feature dropout)
        """
        conv = {}
        for edge in edge_types:
            user_to_item_key = f'{edge}'
            item_to_user_key = f'reverse-{edge}'

            # convolution on user -> item graph
            conv[user_to_item_key] = LightGraphConv(drop_out = drop_out)
            
            # convolution on item -> user graph
            conv[item_to_user_key] = LightGraphConv(drop_out = drop_out)

        self.conv = dgl.nn.pytorch.HeteroGraphConv(conv, aggregate = 'sum')
        self.feature_dropout = nn.Dropout(drop_out)

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        """
        Paramters
        ---------
        graph : dgl.graph
        ufeats, ifeats : torch.FloatTensor
            node features
        ukey, ikey : str
            target node types

        Returns
        -------
        ufeats, ifeats : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ukey : ufeats,
            ikey : ifeats
            }

        out = self.conv(graph, feats)

        ufeats, ifeats = out[ukey], out[ikey]
        ufeats = self.feature_dropout(ufeats)
        ifeats = self.feature_dropout(ifeats)

        return ufeats, ifeats

if __name__ == '__main__':
    from utils import add_degree

    ratings = [1, 2, 3, 4, 5, 6]
    users = torch.tensor([0,0,0,1,1,2,3,4,4,4,2,2]).chunk(len(ratings))
    items = torch.tensor([0,3,5,1,2,4,5,6,0,1,3,5]).chunk(len(ratings))

    graph_data = {}
    for i in range(len(ratings)):
        graph_data[('user', f'{i+1}', 'item')] = (users[i], items[i])
        graph_data[('item', f'reverse-{i+1}', 'user')] = (items[i], users[i])

    g = dgl.heterograph(graph_data)
    add_degree(graph = g, edge_types = ratings)

    n_users, n_items = 5, 7
    ufeats_dim, ifeats_dim = 16, 32
    ufeats = torch.rand(n_users, ufeats_dim)
    ifeats = torch.rand(n_items, ifeats_dim)

    model = LightGCNLayer(edge_types = ratings, drop_out = 0.1)

    ufeats, ifeats = model(g, ufeats, ifeats)
    print(ufeats) # (n_users, out_feats_dim)
    print(ifeats) # (n_items, out_feats_dim)