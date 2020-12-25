import torch
import math
from torch import nn
import networkx as nx
from Components import Policy
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


class DegreePrediction(torch.nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        device = torch.device('cuda:0')
        dtype = torch.float
        self.num_nodes = num_nodes
        self.n_nodes_pow2 = pow(self.num_nodes, 2)
        self.n_nodes_pow3 = pow(self.num_nodes, 3)
        self.weights_t = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, requires_grad=True, device=device, dtype=dtype))
        self.weights_r = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                        requires_grad=True, device=device, dtype=dtype))

    def forward(self, x):
        l2 = x * self.weights_t
        l2_mult = l2.view(self.n_nodes_pow2, 1) * self.weights_r.view(self.n_nodes_pow2, self.n_nodes_pow2)
        y_pred = torch.sum(l2_mult.view(self.num_nodes, self.n_nodes_pow3), dim=1)
        return y_pred


def predit_degree_custom_model(x, y):
    model = DegreePrediction(y.size()[0])
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(1000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return torch.sum(model.weights_t, model.weights_r)
    print(f'prediction : {y_pred}, target:{y}')


if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # predit_degree_custom_model(torch.tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE),
    #                            torch.tensor([1, 1], device=DEVICE, dtype=DTYPE))

    #
    degree_policy = Policy.DegreePolicy()
    edge_lst = [('v0', 'v1'), ('v1', 'v2'), ('v0', 'v2'), ('v2', 'v3')]
    g = nx.Graph(edge_lst).to_directed()
    t_tensor = degree_policy.get_t_tensor(g)
    predit_degree_custom_model(torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=DTYPE,
                                            device=DEVICE), torch.tensor([2, 2, 3, 1], device=DEVICE, dtype=DTYPE))

