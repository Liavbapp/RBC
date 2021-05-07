import datetime

import torch
import torch.nn as nn

from RBC import PowerIteration
from multiprocessing import Pool, TimeoutError
import concurrent.futures
import time
import os
from itertools import product


class NeuralNetworkNodeEmbeddingSourceTargetRouting(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkNodeEmbeddingSourceTargetRouting, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dimensions * 2, 1000),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_nodes ** 2),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res


class NeuralNetworkGraphEmbeddingRbc(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkGraphEmbeddingRbc, self).__init__()
        self.num_nodes = num_nodes
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(dimensions),
            nn.Linear(dimensions, 50),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            # nn.Linear(100, 100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            nn.Linear(50, 50),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(50, num_nodes ** 4),
            nn.Sigmoid(),
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)

        return prop_res.view(prop_res.shape[0], self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)


class NeuralNetworkGraphEmbeddingRouting(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkGraphEmbeddingRouting, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(dimensions),
            nn.Linear(dimensions, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, num_nodes ** 4),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res


class NeuralNetworkNodesEmbeddingRouting(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkNodesEmbeddingRouting, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(num_nodes * dimensions),
            nn.Linear(num_nodes * dimensions, 1000),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 200),
            # nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 5000),
            # nn.BatchNorm1d(5000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(5000, 1000),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, num_nodes ** 4),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res


class EmbeddingNeuralNetwork(nn.Module):
    def __init__(self, dimensions, device, dtype):
        super(EmbeddingNeuralNetwork, self).__init__()
        self.dim = dimensions
        self.drop_out_rate = 0.0
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 100, 2),
            nn.ReLU(),
            # nn.Dropout(self.drop_out_rate),
            nn.Conv2d(100, 200, 2),
            nn.ReLU(),
            # nn.Dropout(self.drop_out_rate),
            nn.Flatten(),
            nn.Linear(200 * 2 * (dimensions - 2), 4096),
            nn.LeakyReLU(),
            # nn.Dropout(self.drop_out_rate),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            # nn.Dropout(self.drop_out_rate),
            nn.Linear(4096, 1),
            nn.LeakyReLU(),
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = x.view(x.shape[0], 4, self.dim)
        x = torch.unsqueeze(x, 1)
        prop_res = self.linear_relu_stack(x).flatten()

        return prop_res


class NisuyNN(nn.Module):
    def __init__(self, dim, num_nodes, device, dtype):
        super(NisuyNN, self).__init__()
        self.device, self.dtype = device, dtype
        self.flatten = nn.Flatten()
        self.num_nodes = num_nodes
        self.embed_dim = dim
        self.pi_handler = PowerIteration.PowerIteration(device=device, dtype=dtype, max_error=0.00001)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim * 2, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_nodes ** 2),
            nn.LeakyReLU(),
            nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, nodes_embeddings, graphs_embeddings, Ts):

        batch_size = len(Ts)
        Cs = torch.full(size=(batch_size, self.num_nodes), fill_value=0.0, dtype=self.dtype, device=self.device)
        for s in (range(self.num_nodes)):
            for t in (range(self.num_nodes)):
                Cs += self.compute_rbc_batch(nodes_embeddings[:, s], nodes_embeddings[:, t], s, graphs_embeddings,
                                             Ts[:, s, t], batch_size)
        return Cs

    def compute_rbc_batch(self, s_embed, t_embed, s_idx, graph_embed, T_vals, batch_size):
        r_policies = self.predicted_policy(s_embed, t_embed, graph_embed, batch_size)
        cs = map(lambda r_policy, t_val: self.accumulate_delta(s_idx, r_policy.squeeze(), t_val), r_policies, T_vals)
        cs = torch.stack(list(cs))
        return cs

    def predicted_policy(self, s, t, graph_embed, batch_size):
        # x = torch.stack([s, t, graph_embed]).view(batch_size, self.embed_dim * 3)
        x = torch.stack([s, t]).view(batch_size, self.embed_dim * 2)
        out = self.linear_relu_stack(x).view(batch_size, self.num_nodes, self.num_nodes).split(1)

        return out

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val):
        new_eigenvalue, eigenvector = self.pi_handler.power_iteration(A=predecessor_prob_matrix)
        normalized_eigenvector = self.noramlize_eiginevector(src, eigenvector, T_val)

        return normalized_eigenvector

    def noramlize_eiginevector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector

#
# class EmbeddingNeuralNetwork(nn.Module):
#     def __init__(self, dimensions, device, dtype):
#         super(EmbeddingNeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(dimensions * 4, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, 1),
#             nn.LeakyReLU()
#         ).to(device=device, dtype=dtype)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         prop_res = self.linear_relu_stack(x).flatten()
#         return prop_res
