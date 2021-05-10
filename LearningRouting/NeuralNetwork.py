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


# class EmbeddingNeuralNetwork(nn.Module):
#     def __init__(self, dimensions, device, dtype):
#         super(EmbeddingNeuralNetwork, self).__init__()
#         self.dim = dimensions
#         self.drop_out_rate = 0.0
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Conv2d(1, 100, 2),
#             nn.ReLU(),
#             # nn.Dropout(self.drop_out_rate),
#             nn.Conv2d(100, 200, 2),
#             nn.ReLU(),
#             # nn.Dropout(self.drop_out_rate),
#             nn.Flatten(),
#             nn.Linear(200 * 2 * (dimensions - 2), 4096),
#             nn.LeakyReLU(),
#             # nn.Dropout(self.drop_out_rate),
#             nn.Linear(4096, 4096),
#             nn.LeakyReLU(),
#             # nn.Dropout(self.drop_out_rate),
#             nn.Linear(4096, 1),
#             nn.LeakyReLU(),
#         ).to(device=device, dtype=dtype)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], 4, self.dim)
#         x = torch.unsqueeze(x, 1)
#         prop_res = self.linear_relu_stack(x).flatten()
#
#         return prop_res
#

class EmbeddingNeuralNetwork(nn.Module):
    def __init__(self, dimensions, device, dtype):
        super(EmbeddingNeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dimensions * 4, dimensions * 2 * 100),
            nn.LeakyReLU(),
            nn.Linear(dimensions * 2 * 100, (dimensions * 2 * 100) * 200),
            nn.LeakyReLU(),
            nn.Linear((dimensions * 2 * 100) * 200, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.LeakyReLU()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x).flatten()
        return prop_res


# class NisuyNN(nn.Module):
#     def __init__(self, dim, num_nodes, device, dtype):
#         super(NisuyNN, self).__init__()
#         self.device, self.dtype = device, dtype
#         self.flatten = nn.Flatten()
#         self.num_nodes = num_nodes
#         self.embed_dim = dim
#         self.pi_handler = PowerIteration.PowerIteration(device=device, dtype=dtype, max_error=0.00001)
#         self.linear_relu_stack = nn.Sequential(
#             nn.Conv2d(1, 100, 2),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear((dim - 1) * 100, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, 4096),
#             nn.LeakyReLU(),
#             nn.Linear(4096, self.num_nodes ** 2),
#             nn.LeakyReLU(),
#             nn.Sigmoid()
#         ).to(device=device, dtype=dtype)

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
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.num_nodes ** 2),
            nn.LeakyReLU(),
            nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, batch_node_embeddings, batch_graphs_embeddings, batch_Ts, mult_const_batch, add_const_batch):
        """
        The forward pass of the network
        :param batch_node_embeddings: nodes embeddings tensor, dimensions: batch_size x nodes x nodes x node_embed_dim
        :param batch_graphs_embeddings: graphs embeddings tensor, dimensions: batch_size x graph_embed_dim
        :param batch_Ts: Traffic matrix for each graph in the batch, dimensions: batch_size x nodes x nodes
        :return rbcs_prediction: the predicted values of the rbcs, dimensions: batch_size x nodes
        """
        self.mult_const, self.add_const = mult_const_batch, add_const_batch
        uv_tensor = self.create_uv_tensor(batch_node_embeddings)
        rbcs_prediction = self.rbc_prediction(uv_tensor, batch_node_embeddings, batch_Ts, batch_size=len(batch_Ts))

        return rbcs_prediction

    def create_uv_tensor(self, batch_node_embeddings):
        """
        creating tensor of cartesian product of the uv nodes
        :param batch_node_embeddings: node embedding of the whole batch
        :return batch_uv_tensor: dimensions: (batch_size * nodes x nodes x (2 * embed_dim))
        """
        batch_uv_tensor = map(lambda node_embeddings: self.single_uv_tensor(node_embeddings), batch_node_embeddings)
        batch_uv_tensor = torch.stack(list(batch_uv_tensor))

        return batch_uv_tensor

    def single_uv_tensor(self, node_embeddings):
        """
        compute the uv tensor of single batch instance
        :param node_embeddings: nodes embeddings of the current instance, dimensions: nodes x dimensions
        :return uv:
        """
        uv = [torch.cartesian_prod(node_embeddings.T[i], node_embeddings.T[i]) for i in range(0, self.embed_dim)]
        uv = torch.stack(uv).transpose(0, 2).reshape(self.num_nodes, self.num_nodes, 2 * self.embed_dim)
        return uv

    def rbc_prediction(self, uv_batch, node_embeddings_batch, Ts_batch, batch_size):
        """
        predicting the rbc for each instance in the batch
        :param uv_batch: the uv_tensor of the whole batch, dimensions: batch_size x nodes x nodes x (2 * embed_dim)
        :param node_embeddings_batch: dimensions: batch_size x nodes x embed_dim
        :param Ts_batch: the Traffic tensor of the graphs in the batch, dimensions: batch_size x nodes x nodes
        :param batch_size:
        :return delta: the predicted rbcs values for the whole batch, dimension: batch_size x nodes
        """
        deltas = torch.full(size=(batch_size, self.num_nodes), fill_value=0.0, dtype=self.dtype, device=self.device)
        for s in (range(self.num_nodes)):
            for t in (range(self.num_nodes)):
                embedding_s, embedding_t, s_idx = node_embeddings_batch[:, s], node_embeddings_batch[:, t], s
                uv_st, Ts_st = uv_batch[:, s, t], Ts_batch[:, s, t]
                deltas += self.compute_delta_st(uv_batch, embedding_s, embedding_t, s_idx, Ts_st, batch_size)
        return deltas

    def compute_delta_st(self, uv_batch, s_embed, t_embed, s_idx, T_st, batch_size):
        """
        compute the delta vector of the [source, target] pair, of the whole batch
        :param uv_batch: uv_tensor of the [source, target] pair, dimension: batch_size x (2 * embedding_dim)
        :param s_embed: source node embedding vector of the whole batch, dimensions: batch_size x embed_dim
        :param t_embed: target node embedding vector of the whole batch, dimensions: batch_size x embed_dim
        :param s_idx: index of the source node
        :param T_st: Traffic_tensor[source, target] for each graph in batch, dimensions: batch_size x 1
        :param batch_size:
        :return delta: the delta vector for the source,target pair, dimensions: batch_size x nodes
        """
        r_policies = self.predict_st_policy(uv_batch, s_embed, t_embed, batch_size)
        delta = map(lambda r_policy, t_val: self.accumulate_delta(s_idx, r_policy.squeeze(), t_val), r_policies, T_st)
        delta = torch.stack(list(delta))
        return delta

    def predict_st_policy(self, uv_batch, embed_s, embed_t, batch_size):
        """
        predicting the routing policy of the [source, target] pair
        :param uv_batch: uv_tensor of the [source, target] pair, dimension: batch_size x (2 * embedding_dim)
        :param embed_s: embedding vector of source node, of the whole batch, dimensions: batch_size x embed_dim
        :param embed_t:  embedding vector of target node, of the whole batch, dimensions: batch_size x embed_dim
        :param batch_size:
        :return:
        """
        x = torch.stack([embed_s, embed_t]).view(batch_size, self.embed_dim * 2)
        # x = torch.stack([embed_s, embed_t]).view(batch_size, 2, self.embed_dim).unsqueeze(dim=1)  # conv_layers
        # out = self.linear_relu_stack(x).view(batch_size, self.num_nodes, self.num_nodes).split(1)
        # return out

        out = self.linear_relu_stack(x).view(batch_size, self.num_nodes, self.num_nodes)

        return (torch.mul(out, self.mult_const) + self.add_const).split(1)

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val):
        new_eigenvalue, eigenvector = self.pi_handler.power_iteration(A=predecessor_prob_matrix)
        normalized_eigenvector = self.noramlize_eiginevector(src, eigenvector, T_val)

        return normalized_eigenvector

    def noramlize_eiginevector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector
