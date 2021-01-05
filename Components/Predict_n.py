import datetime
import math

import torch
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
        layer2 = (x * self.weights_t).view(self.num_nodes, self.num_nodes, 1, 1) * r_const
        weights_r_comb = torch.mul(self.weights_r, r_zeros) + r_const
        all_delta_arrays = [self.accumulate_delta(s, weights_r_comb[s, t].clone().detach(), layer2[s, t, s, s]) for s in
                            range(0, len(x)) for t in range(0, len(x))]
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)

        # layer3 = torch.mul(layer2.view(self.n_nodes_pow2, 1), weights_r_comb.view(self.n_nodes_pow2, self.n_nodes_pow2))
        # layer3 = layer3.view(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)
        # y_pred = torch.sum(layer3, 1).sum(0).sum(1)
        return rbc_arr

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val):
        eigenvalues, eigenvectors = torch.eig(input=predecessor_prob_matrix, eigenvectors=True)
        eigenvector = self.get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, torch.tensor([[1.0, 0.0]],
                                                                                                 device=DEVICE,
                                                                                                 dtype=DTYPE))
        eigenvector = self.compute_eigenvector_values(src, eigenvector, T_val)
        return eigenvector

    def compute_eigenvector_values(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector

    def get_eigenvector_by_eigenvalue(self, eigenvalues, eigenvectors, eigenvalue):
        comparison_sum = (eigenvalues == eigenvalue).sum(dim=1)  # pair-wise compare and sum how many elements equal
        indicies = torch.nonzero(comparison_sum == eigenvalues.size(1))  # the equal indexes
        eigenvalue_idx = int(indicies[0][0])  # extract the index
        eigenvector = eigenvectors[:, eigenvalue_idx]  # extract the eigenvector col corresponding to eigenvalue index
        return eigenvector


def predict_degree_custom_model(graph, adj_mat, y):
    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    zero_mat, const_mat = get_fixed_mat(adj_mat, graph)

    start_time = datetime.datetime.now()
    for t in range(100):
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


def get_fixed_mat(adj_mat, graph):
    nodes = list(graph.nodes)
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            edges_path_lst = list(nx.all_simple_edge_paths(graph, nodes[s], nodes[t]))
            edges = set()
            if len(edges_path_lst) > 0:
                # edges = set(sum(edges_path_lst, []))
                edges = set(edges_path_lst[0])
            constant_mat[s, t, s, s] = 1
            for edge in edges:
                zeros_mat[s, t, edge[1], edge[0]] = 1
    return zeros_mat, constant_mat


if __name__ == '__main__':
    degree_policy = Policy.DegreePolicy()
    edge_lst = [(0, 1), (1, 2), (0, 2)]
    g = nx.Graph(edge_lst)
    adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    target_matrix = torch.tensor(list(map(float, dict(nx.degree(g)).values())), device=DEVICE, dtype=DTYPE)
    g = g.to_directed()
    t_model, r_model = predict_degree_custom_model(g, adj_matrix, target_matrix)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)
