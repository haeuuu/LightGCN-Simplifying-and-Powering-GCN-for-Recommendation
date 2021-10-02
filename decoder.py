import torch
import torch.nn as nn

import dgl
import dgl.function as fn


class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        ufeats : torch.FloatTensor
            user features
        ifeats : torch.FloatTensor
            item features

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_users, 1)
        """

        with graph.local_scope():
            graph.nodes[ukey].data['h'] = ufeats
            graph.nodes[ikey].data['h'] = ifeats
            graph.apply_edges(fn.u_dot_v('h', 'h', 'r'))
            pred = graph.edata['r']

        return pred

class BilinearDecoder(nn.Module):
    def __init__(self,
                feats_dim,
                n_classes,
                n_basis = None):
        super().__init__()
        """Bilinear decoder for link prediction

        Parameters
        ----------
        feats_dim : int
            dimension of input features
        n_classes : int
            number of edge types
        n_basis : int
            number of basis ( <= n_classes )

        Notes
        -----
        if not weight_sharing:
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        else:
            score_{i, j, s} = ufeats_{i} @ P_s @ ifeats_{j} where s <= r
            logits_{i, j, r} = a_{r, s = 1} * score_{i, j, s = 1} + ...
            
            Q_{r} = \sum_{s = 1} a_{r,s} * P_{s}
        """
        if n_basis is not None:
            # weight sharing
            self.to_Q_r = nn.Linear(n_basis, n_classes)
        else:
            n_basis = n_classes
            self.to_Q_r = nn.Identity()

        self.P_r = nn.ModuleList()
        for _ in range(n_basis):
            self.P_r.append(nn.Linear(feats_dim, feats_dim))

    def reset_parameters(self):
        for p_r in self.P_r:
            torch.nn.init.xavier_uniform_(p_r.weight)

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        """
        Paramters
        ---------
        graph : dgl.homograph
            item -> user graph or user -> item graph
        ufeats, ifeats : torch.FloatTensor

        Returns
        -------
        pred_edge_types : torch.FloatTensor
            shape : (n_users, n_classes)
        """

        with graph.local_scope():
            graph.nodes[ikey].data['h'] = ifeats

            pred_scores = []
            for P_r in self.P_r:
                graph.nodes[ukey].data['h'] = P_r(ufeats)
                graph.apply_edges(fn.u_dot_v('h', 'h', 'r'))

                pred = graph.edata['r']
                pred_scores.append(pred)

            pred_edge_types = self.to_Q_r(torch.cat(pred_scores, dim = -1))

        return pred_edge_types

if __name__ == '__main__':
    n_ratings = 6
    n_users, n_items = 5, 7
    users = torch.tensor([0,0,0, 1,1, 2, 3, 4,4,4,4,4])
    items = torch.tensor([0,3,5, 1,2, 4, 5, 0,1,3,5,6])
    
    output_feats_dim = 16 # GAE output features
    ufeats = torch.rand(n_users, output_feats_dim)
    ifeats = torch.rand(n_items, output_feats_dim)

    dec_graph_data = {
        ('item', 'reverse-rating', 'user') : (items, users)
        }
    dec_g = dgl.heterograph(dec_graph_data)

    # example 1. Bilinear Decoder
    decoder = BilinearDecoder(feats_dim = output_feats_dim,
                            n_classes = 1,
                            n_basis = None)

    logits = decoder(dec_g, ufeats, ifeats)
    print(logits) # (n_edges, n_classes)

    # example 2. DotProduct Decoder
    decoder = DotDecoder()
    logits = decoder(dec_g, ufeats, ifeats)
    print(logits) # (n_pos, n_classes)