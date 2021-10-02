import torch
import torch.nn as nn

from lightgcn import LightGCNLayer
from decoder import DotDecoder, BilinearDecoder


class LightGCN(nn.Module):
    def __init__(self,
                n_layers,
                edge_types,
                drop_out,
                feats_dim,
                n_basis = None,
                learnable_weight = False):
        super().__init__()
        """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
        paper : https://arxiv.org/pdf/2002.02126.pdf

        n_layers : int
            number of GCMC layers
        edge_types : list
            all edge types
        drop_out : float
            dropout rate (neighbors)
        learnable_weight : boolean
            whether to learn weights for embedding aggregation
            if False, use 1/n_layers
        """
        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(LightGCNLayer(edge_types = edge_types,
                                                drop_out = drop_out))
        if learnable_weight:
            self.weights = nn.Parameter(torch.ones(n_layers)/self.n_layers)
        else:
            self.weights = torch.ones(n_layers)/self.n_layers
            
        # self.decoder = DotDecoder()
        self.decoder = BilinearDecoder(feats_dim = feats_dim,
                                        n_classes = len(edge_types),
                                        n_basis = n_basis)

    def encode(self, graph, ufeats, ifeats, ukey, ikey):
        u_hidden, i_hidden = ufeats, ifeats
        for w, encoder in zip(self.weights, self.encoders):
            ufeats, ifeats = encoder(graph, ufeats, ifeats, ukey, ikey)
            u_hidden += w * ufeats
            i_hidden += w * ifeats

        return ufeats, ifeats

    def decode(self, pos_graph, neg_graph, ufeats, ifeats, ukey, ikey):
        pred_pos = self.decoder(pos_graph, ufeats, ifeats, ukey, ikey)
        pred_neg = self.decoder(neg_graph, ufeats, ifeats, ukey, ikey)

        return pred_pos, pred_neg

    def forward(self,
                enc_graph,
                dec_graph,
                ufeats,
                ifeats,
                ukey = 'user',
                ikey = 'item'):
        """
        Parameters
        ----------
        enc_graph : dgl.graph
        dec_graph : dgl.homograph

        Notes
        -----
        1. LightGCN encoder
            1 ) message passing
                MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
            2 ) aggregation
                \sum_{j \in N(i), r} MP_{j -> i, r}

        2. final features
            user_{i} = mean( h_{i, layer_1}, h_{i, layer_2}, ... )
            item_{j} = mean( h_{j, layer_1}, h_{j, layer_2}, ... )

        3. Bilinear decoder
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        """

        ufeats, ifeats = self.encode(enc_graph, ufeats, ifeats, ukey, ikey)
        pred_edge_types = self.decoder(dec_graph, ufeats, ifeats, ukey, ikey)

        return pred_edge_types