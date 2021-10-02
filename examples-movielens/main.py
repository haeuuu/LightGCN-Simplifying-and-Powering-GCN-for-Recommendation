import sys
sys.path.append('..')

import os
import time
import fire

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import LightGCN
from data import MovieLens


class Trainer:
    def __init__(self,
                data_name = 'ml-100k',
                valid_ratio = 0.1,
                test_ratio = 0.1
                ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = MovieLens(name = data_name,
                                test_ratio = test_ratio,
                                valid_ratio = valid_ratio,
                                device = device)

    def train(self,
            n_layers = 1,
            hidden_feats_dim = 500,
            out_feats_dim = 75,
            agg = 'stack',
            drop_out = 0.7,
            activation = 'leaky',
            n_basis = 2,
            lr = 0.01,
            iteration = 2000,
            log_interval = 1,
            early_stopping = 100,
            lr_intialize_step = 50,
            lr_decay = 0.5,
            train_min_lr = 0.001
            ):

        n_users, n_items = self.dataset.user_feature.shape[0], self.dataset.movie_feature.shape[0]
        self.dataset.user_feature = nn.Embedding(n_users, out_feats_dim)
        self.dataset.movie_feature = nn.Embedding(n_items, out_feats_dim)

        model = LightGCN(n_layers = n_layers,
                        edge_types = self.dataset.possible_rating_values,
                        drop_out = drop_out,
                        feats_dim = out_feats_dim,
                        n_basis = n_basis)
        print(model)

        device = self.dataset._device
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        possible_edge_types = torch.FloatTensor(self.dataset.possible_rating_values).unsqueeze(0).to(device)

        train_gt_labels = self.dataset.train_labels
        train_gt_ratings = self.dataset.train_truths

        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        count_rmse = 0
        count_num = 0
        count_loss = 0

        self.dataset.train_enc_graph = self.dataset.train_enc_graph.int().to(device)
        self.dataset.train_dec_graph = self.dataset.train_dec_graph.int().to(device)
        self.dataset.valid_enc_graph = self.dataset.train_enc_graph
        self.dataset.valid_dec_graph = self.dataset.valid_dec_graph.int().to(device)
        self.dataset.test_enc_graph = self.dataset.test_enc_graph.int().to(device)
        self.dataset.test_dec_graph = self.dataset.test_dec_graph.int().to(device)

        print(f"Start training on {device}...")
        for iter_idx in range(iteration):
            model.train()
            logits = \
                model(self.dataset.train_enc_graph,
                    self.dataset.train_dec_graph,
                    self.dataset.user_feature(torch.arange(n_users)),
                    self.dataset.movie_feature(torch.arange(n_items)))
            loss = criterion(logits, train_gt_labels).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # pred_{i,j} = \sum_{r = 1} r * P(link_{i,j} = r)
            pred_ratings = (torch.softmax(logits, dim=1) * possible_edge_types).sum(dim=1)
            rmse = ((pred_ratings - train_gt_ratings) ** 2).sum()
            count_rmse += rmse.item()
            count_num += pred_ratings.shape[0]

            if iter_idx and iter_idx % log_interval == 0:
                log = f"[{iter_idx}/{iteration}-iter] | [train] loss : {count_loss/iter_idx:.4f}, rmse : {count_rmse/count_num:.4f}"
                count_rmse, count_num = 0, 0

            if iter_idx and iter_idx % (log_interval*10) == 0:
                valid_rmse = self.evaluate(model, self.dataset, possible_edge_types, n_users, n_items, data_type = 'valid')
                log += f" | [valid] rmse : {valid_rmse:.4f}"

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    best_test_rmse = self.evaluate(model, self.dataset, possible_edge_types, n_users, n_items, data_type = 'test')
                    log += f" | [test] rmse : {best_test_rmse:.4f}"

                    torch.save(model, './model.pt')

                else:
                    no_better_valid += 1
                    if no_better_valid > early_stopping:
                        print("Early stopping threshold reached. Stop training.")
                        break
                    if no_better_valid > lr_intialize_step:
                        new_lr = max(lr * lr_decay, train_min_lr)
                        if new_lr < lr:
                            lr = new_lr
                            print("\tChange the LR to %g" % new_lr)
                            for p in optimizer.param_groups:
                                p['lr'] = lr
                            no_better_valid = 0

            if iter_idx and iter_idx  % log_interval == 0:
                print(log)

        print(f'[END] Best Iter : {best_iter} Best Valid RMSE : {best_valid_rmse:.4f}, Best Test RMSE : {best_test_rmse:.4f}')

    def evaluate(self, model, dataset, possible_edge_types, n_users, n_items, data_type = 'valid'):
        if data_type == "valid":
            rating_values = dataset.valid_truths
            enc_graph = dataset.valid_enc_graph
            dec_graph = dataset.valid_dec_graph
        elif data_type == "test":
            rating_values = dataset.test_truths
            enc_graph = dataset.test_enc_graph
            dec_graph = dataset.test_dec_graph

        model.eval()
        with torch.no_grad():
            logits = model(enc_graph, dec_graph,
                            dataset.user_feature(torch.arange(n_users)), dataset.movie_feature(torch.arange(n_items)))
            pred_ratings = (torch.softmax(logits, dim=1) * possible_edge_types).sum(dim=1)
            rmse = ((pred_ratings - rating_values) ** 2.).mean().item()
            rmse = np.sqrt(rmse)
            
        return rmse

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    fire.Fire(Trainer)
