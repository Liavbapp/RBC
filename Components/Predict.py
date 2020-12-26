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
    def __init__(self, num_nodes):
        super().__init__()
        device = torch.device('cuda:0')
        dtype = torch.float
        self.num_nodes = num_nodes
        self.n_nodes_pow2 = pow(self.num_nodes, 2)
        self.n_nodes_pow3 = pow(self.num_nodes, 3)

        # indicies = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1]]
        self.weights_t_fix = torch.randn(self.num_nodes, self.num_nodes, requires_grad=False, device=device,
                                         dtype=dtype)
        self.weights_t_fix[0, 0] = 0
        self.weights_t_fix[1, 1] = 0


        self.mask_t = torch.zeros(2, 2, requires_grad=False, device=device, dtype=dtype)
        self.mask_t[0, 0] = 1
        self.mask_t[1, 1] = 1


        self.weights_t = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, requires_grad=True, device=device, dtype=dtype))

        self.fix_weights_r = torch.rand(2, 2, 2, 2, requires_grad=False, device=device, dtype=dtype)
        self.fix_weights_r[0, 0, 0, 0] = 1
        self.fix_weights_r[0, 1, 0, 0] = 1
        self.fix_weights_r[1, 0, 1, 1] = 1
        self.fix_weights_r[1, 1, 1, 1] = 1
        self.fix_weights_r[0, 0, 0, 1] = 0
        self.fix_weights_r[0, 0, 1, 0] = 0
        self.fix_weights_r[0, 0, 1, 1] = 0
        self.fix_weights_r[0, 1, 0, 1] = 0
        self.fix_weights_r[0, 1, 1, 1] = 0
        self.fix_weights_r[1, 0, 0, 0] = 0
        self.fix_weights_r[1, 0, 1, 0] = 0
        self.fix_weights_r[1, 1, 0, 0] = 0
        self.fix_weights_r[1, 1, 0, 1] = 0
        self.fix_weights_r[1, 1, 1, 0] = 0

        # for idx in indicies:
        #     self.fix_weights_r[idx] = 1
        self.mask_r = torch.zeros(2, 2, 2, 2, requires_grad=False, device=device, dtype=dtype)
        self.mask_r[0, 0, 0, 0] = 1
        self.mask_r[0, 1, 0, 0] = 1
        self.mask_r[1, 0, 1, 1] = 1
        self.mask_r[1, 1, 1, 1] = 1
        self.mask_r[0, 0, 0, 1] = 1
        self.mask_r[0, 0, 1, 0] = 1
        self.mask_r[0, 0, 1, 1] = 1
        self.mask_r[0, 1, 0, 1] = 1
        self.mask_r[0, 1, 1, 1] = 1
        self.mask_r[1, 0, 0, 0] = 1
        self.mask_r[1, 0, 1, 0] = 1
        self.mask_r[1, 1, 0, 0] = 1
        self.mask_r[1, 1, 0, 1] = 1
        self.mask_r[1, 1, 1, 0] = 1

        # for idx in indicies:
        #     self.mask[idx] = 1

        self.weights_r = torch.nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                        requires_grad=True, device=device, dtype=dtype))

    def forward(self, x):
        weights_t_comb = self.mask_t * self.weights_t_fix + (1 - self.mask_t) * self.weights_t
        l2 = x * weights_t_comb
        weights_r_comb = self.mask_r * self.fix_weights_r + (1 - self.mask_r) * self.weights_r
        l2_mult = l2.view(self.n_nodes_pow2, 1) * weights_r_comb.view(self.n_nodes_pow2, self.n_nodes_pow2)
        # y_pred = torch.sum(torch.sum(l2_mult, dim=1).flatten(1))
        l2_mult = l2_mult.view(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)
        y_pred = torch.sum(l2_mult, 1).sum(0).sum(1)
        return y_pred


def predit_degree_custom_model(x, y):
    model = DegreePrediction(y.size()[0])
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(4000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.weights_t.copy_(model.weights_t.data.clamp(min=0))
            model.weights_r.copy_(model.weights_r.data.clamp(min=0))

    print(f'prediction : {y_pred}, target:{y}')
    model_t = model.mask_t * model.weights_t_fix + (1 - model.mask_t) * model.weights_t
    model_r = model.mask_r * model.fix_weights_r + (1 - model.mask_r) * model.weights_r
    return model_t, model_r


if __name__ == '__main__':
    degree_policy = Policy.DegreePolicy()
    edge_lst = [('v0', 'v1')]
    g = nx.Graph(edge_lst).to_directed()
    t_tensor = degree_policy.get_t_tensor(g)
    t_weights, r_weights = predit_degree_custom_model(torch.tensor([[0, 1], [1, 0]], dtype=DTYPE, device=DEVICE),
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
