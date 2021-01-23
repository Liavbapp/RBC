import datetime
import random

import torch
import networkx as nx
import Components.RBC as RBC
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


class DegreePrediction(torch.nn.Module):
    def __init__(self, adj_mat):
        super().__init__()
        self.num_nodes = adj_mat.size()[0]
        self.weights_t = torch.nn.Parameter(
            torch.rand(self.num_nodes, self.num_nodes, requires_grad=True, device=DEVICE, dtype=DTYPE))
        self.weights_r = torch.nn.Parameter(torch.rand(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                                                       requires_grad=True, device=DEVICE, dtype=DTYPE))

    def forward(self, x, r_zeros, r_const):
        layer2 = (x * self.weights_t).view(self.num_nodes, self.num_nodes, 1, 1) * r_const
        weights_r_comb = torch.mul(self.weights_r, r_zeros) + r_const
        all_delta_arrays = [self.accumulate_delta(s, weights_r_comb[s, t], layer2[s, t, s, s]) for s in
                            range(0, len(x)) for t in range(0, len(x))]
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
        return rbc_arr

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val):
        new_eigenevalue, eiginevector = self.power_iteration(predecessor_prob_matrix)
        eigenvector2 = eiginevector
        eigenvector = self.compute_eigenvector_values(src, eigenvector2, T_val)
        return eigenvector

    def compute_eigenvector_values(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector

    def eigenvalue(self, A, v):
        Av = torch.mm(A, v)
        return torch.dot(v.flatten(), Av.flatten())

    def power_iteration(self, A, num_iter=5):
        n, d = A.shape
        v = (torch.ones(d) / np.sqrt(d)).to(device=DEVICE, dtype=DTYPE).view(d, 1)
        ev = self.eigenvalue(A, v)

        i = 0
        while True:
            i += 1
            Av = torch.mm(A, v)
            v_new = Av / torch.linalg.norm(Av.flatten())

            ev_new = self.eigenvalue(A, v_new)
            if torch.abs(ev - ev_new) < 0.0001:
                break

            v = v_new
            ev = ev_new

        return ev_new, v_new.flatten()


def predict_degree_custom_model(adj_mat, y):
    zero_mat, const_mat = get_fixed_mat(adj_mat)

    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    start_time = datetime.datetime.now()
    for t in range(500):
        y_pred = model(adj_mat, zero_mat, const_mat)
        loss = criterion(y_pred, y)
        if t % 20 == 0 or t == 0:
            print(t, loss.item())
        # print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'run time: {datetime.datetime.now() - start_time} , prediction : {y_pred}, target:{y}')
    model_t = model.weights_t * adj_mat
    model_r = model.weights_r * zero_mat + const_mat
    return model_t, model_r


def get_fixed_mat(adj_mat):
    adj_matrix_t = adj_mat.t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1  # every src,src in policy matrix must have 1 to have a free variable
            zeros_mat[s, t] = adj_matrix_t  # putting zeros where the transposed adj_matrix has zeros
            # zeros_mat[s, t, s] = 0  # we want to zeros all edges that go into the source node

    return zeros_mat, constant_mat


def test_degree():
    # edge_lst = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    g = nx.gnp_random_graph(10, 0.5, directed=True)

    g = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])

    # edge_lst = [(0, 3), (1, 4), (2, 3), (2, 5), (3, 2), (3, 0), (3, 5), (4, 5), (4, 1), (5, 4), (5, 3), (5, 2)]
    # g = nx.DiGraph(edge_lst)
    # g = nx.watts_strogatz_graph(n=6, k=2, p=0.5)
    # g = g.to_directed()


    nodes_mapping_reverse = {k: v for k, v in enumerate(list(g.nodes()))}
    # adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    # target_matrix =  torch.tensor(list(map(float, dict(nx.degree(g)).values())), device=DEVICE, dtype=DTYPE)
    adj_matrix = get_adj_mat(g, nodes_mapping_reverse)
    target_matrix = get_degree_in_out(adj_matrix)
    t_model, r_model = predict_degree_custom_model(adj_matrix, target_matrix)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)

def test_betweenness():
    # edge_lst = [(0, 1), (1, 2), (2, 0), (2, 3)]
    # g = nx.DiGraph(edge_lst)
    g = nx.gnp_random_graph(10, 0.5, directed=True)
    g = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])

    # adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    nodes_mapping_reverse = {k: v for k, v in enumerate(list(g.nodes()))}
    adj_matrix = get_adj_mat(g, nodes_mapping_reverse)
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    betweenness_tensor = get_betweenness_tensor(g, nodes_mapping)
    t_model, r_model = predict_degree_custom_model(adj_matrix, betweenness_tensor)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)

def get_degree_in_out(adj_matrix):
    row_sum = torch.sum(adj_matrix, dim=0)
    col_sum = torch.sum(adj_matrix, dim=1)
    in_out_degree = row_sum + col_sum
    return in_out_degree


def get_betweenness_tensor(g, nodes_mapping):
    tensor_raw = torch.tensor(list(nx.betweenness_centrality(g).values()), dtype=DTYPE, device=DEVICE)
    tensor_norm = tensor_raw.clone()
    for node_val, node_idx in nodes_mapping.items():
        tensor_norm[node_val] = tensor_raw[node_idx]

    return tensor_norm



def get_adj_mat(g, nodes_mapping):
    adj_matrix_raw = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    adj_matrix_norm = torch.clone(adj_matrix_raw)
    for s in range(0, g.number_of_nodes()):
        for t in range(0, g.number_of_nodes()):
            adj_matrix_norm[nodes_mapping[s], nodes_mapping[t]] = adj_matrix_raw[s, t]
    # for node_val in g.nodes():
    #     node_index = nodes_mapping[node_val]
    #     adj_matrix_norm[:, node_val] = adj_matrix_raw[:, node_index]
    #     adj_matrix_norm[node_val, :] = adj_matrix_raw[node_index, :]

    return adj_matrix_norm







if __name__ == '__main__':
    # test_degree()
    test_betweenness()