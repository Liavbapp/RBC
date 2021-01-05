import datetime

import torch
import math
from torch import nn
import networkx as nx
from Components import Policy
import numpy as np
import Components.RBC as RBC

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


class DegreePrediction(torch.nn.Module):
    def __init__(self, adj_mat):
        super().__init__()
        device = torch.device('cuda:0')
        dtype = torch.float
        self.num_nodes = adj_mat.size()[0]
        self.n_nodes_pow2 = pow(self.num_nodes, 2)
        self.n_nodes_pow3 = pow(self.num_nodes, 3)
        self.weights_t = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, requires_grad=True, device=device, dtype=dtype))

        self.weights_r = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                        requires_grad=True, device=device, dtype=dtype))

    def forward(self, x, r_zeros, r_const):
        layer2 = x * self.weights_t
        weights_r_comb = torch.mul(self.weights_r, r_zeros) + r_const
        layer3 = torch.mul(layer2.view(self.n_nodes_pow2, 1), weights_r_comb.view(self.n_nodes_pow2, self.n_nodes_pow2))
        layer3 = layer3.view(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)
        y_pred = torch.sum(layer3, 1).sum(0).sum(1)
        return y_pred


def predict_degree_custom_model(adj_mat, y):
    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    zero_mat, const_mat = get_fixed_mat(adj_mat)

    start_time = datetime.datetime.now()
    for t in range(300):
        y_pred = model(adj_mat, zero_mat, const_mat)
        loss = criterion(y_pred, y)
        if t % 20 == 0:
            print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'run time: {datetime.datetime.now() - start_time} , prediction : {y_pred}, target:{y}')
    model_t = model.weights_t * adj_mat
    model_r = model.weights_r * zero_mat + const_mat
    return model_t, model_r


def get_fixed_mat(adj_mat):
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1
            if adj_mat[s, t]:
                zeros_mat[s, t, t, s] = 1
    return zeros_mat, constant_mat


if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # t_weights, r_weights = predit_degree_custom_model(
    #     torch.tensor([[0, 1], [1, 0]], requires_grad=True, dtype=DTYPE, device=DEVICE),
    #     torch.tensor([1, 1], device=DEVICE, dtype=DTYPE))
    #
    # t_model = t_weights.to(device=torch.device("cpu"))
    # r_model = r_weights.to(device=torch.device("cpu"))
    # rbc_pred = RBC.rbc(g, r_model, t_model)
    # print(rbc_pred)

    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2'), ('v2', 'v3')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # t_model, r_model = predit_degree_custom_model(
    #     torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=DTYPE,
    #                  device=DEVICE, requires_grad=True), torch.tensor([2, 2, 3, 1], device=DEVICE, dtype=DTYPE))
    # t_model = t_model.to(device=torch.device("cpu"))
    # r_model = r_model.to(device=torch.device("cpu"))
    # rbc_pred = RBC.rbc(g, r_model, t_model)
    # print(rbc_pred)

    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2'), ('v2', 'v3')]
    # g = nx.Graph(edge_lst)
    # adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    # target_matrix = torch.tensor(list(map(float,dict(nx.degree(g)).values())), device=DEVICE, dtype=DTYPE)
    # g = g.to_directed()
    # t_model, r_model = predit_degree_custom_model(adj_matrix, target_matrix)
    # t_model = t_model.to(device=torch.device("cpu"))
    # r_model = r_model.to(device=torch.device("cpu"))
    # rbc_pred = RBC.rbc(g, r_model, t_model)
    # print(rbc_pred)

    degree_policy = Policy.DegreePolicy()
    g = nx.watts_strogatz_graph(n=100, k=8, p=0.5)
    adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    target_matrix = torch.tensor(list(map(float, dict(nx.degree(g)).values())), device=DEVICE, dtype=DTYPE)
    g = g.to_directed()
    t_model, r_model = predict_degree_custom_model(adj_matrix, target_matrix)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)
