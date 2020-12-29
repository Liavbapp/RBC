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
        weights_r_comb = self.weights_r * r_zeros + r_const
        l2_mult = layer2.view(self.n_nodes_pow2, 1) * weights_r_comb.view(self.n_nodes_pow2, self.n_nodes_pow2)
        l2_mult = l2_mult.view(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)
        y_pred = torch.sum(l2_mult, 1).sum(0).sum(1)
        return y_pred


def predit_degree_custom_model(adj_mat, y):
    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    zero_mat, const_mat = get_fixed_mat(adj_mat)

    for t in range(4000):
        y_pred = model(adj_mat, zero_mat, const_mat)
        loss = criterion(y_pred, y)
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     model.weights_t.copy_(model.weights_t.data.clamp(min=0))
        #     model.weights_r.copy_(model.weights_r.data.clamp(min=0))

    print(f'prediction : {y_pred}, target:{y}')
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
    degree_policy = Policy.DegreePolicy()
    edge_lst = [('v0', 'v1')]
    g = nx.Graph(edge_lst).to_directed()
    t_tensor = degree_policy.get_t_tensor(g)
    t_weights, r_weights = predit_degree_custom_model(
        torch.tensor([[0, 1], [1, 0]], requires_grad=True, dtype=DTYPE, device=DEVICE),
        torch.tensor([1, 1], device=DEVICE, dtype=DTYPE))

    t_model = t_weights.to(device=torch.device("cpu"))
    r_model = r_weights.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)

    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2'), ('v2', 'v3')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # t_model, r_model = predit_degree_custom_model(
    #     torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=DTYPE,
    #                  device=DEVICE), torch.tensor([2, 2, 3, 1], device=DEVICE, dtype=DTYPE))
    # t_model = t_model.to(device=torch.device("cpu"))
    # r_model = r_model.to(device=torch.device("cpu"))
    # rbc_pred = RBC.rbc(g, r_model, t_model)
    # print(rbc_pred)
