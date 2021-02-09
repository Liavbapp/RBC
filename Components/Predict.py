import datetime
import random

import torch
import networkx as nx
import Components.RBC as RBC
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


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

    return adj_matrix_norm


def get_fixed_mat(adj_mat, reverse_nodes_mapping, g):
    adj_matrix_t = adj_mat.t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    path_exist = torch.full(size=(adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1  # every src,src in policy matrix must have 1 to have a free variable
            zeros_mat[s, t] = adj_matrix_t  # putting zeros where the transposed adj_matrix has zeros

            # todo: instead of the code below write algorithm that for a given adj_matrix finds if there is a path between s,t (path len must be >= 1, that mean no self-loops)
            if s == t:
                try:
                    nx.find_cycle(g, reverse_nodes_mapping[s])
                    path_exist[s, t] = 1.0
                except:
                    path_exist[s, t] = 0.0
            else:
                if nx.has_path(g, reverse_nodes_mapping[s], reverse_nodes_mapping[t]):
                    path_exist[s, t] = 1.0
                else:
                    path_exist[s, t] = 0.0

    return zeros_mat, constant_mat, path_exist


class DegreePrediction(torch.nn.Module):
    def __init__(self, adj_mat):
        super().__init__()
        self.num_nodes = adj_mat.size()[0]
        self.weights_t = torch.nn.Parameter(
            torch.rand(self.num_nodes, self.num_nodes, requires_grad=True, device=DEVICE, dtype=DTYPE))
        self.weights_r = torch.nn.Parameter(torch.rand(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                                                       requires_grad=True, device=DEVICE, dtype=DTYPE))

    def forward(self, x, r_zeros, r_const, t_paths):
        weights_t_fixed = self.weights_t * t_paths
        weights_r_comb = torch.mul(self.weights_r,
                                   r_zeros) + r_const  # todo: I found that if I ommit r_const the power iteration doesnt converge!!!

        all_delta_arrays = [self.accumulate_delta(s, weights_r_comb[s, t], weights_t_fixed[s, t]) for s in
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

    def power_iteration(self, A, num_iter=10):
        n, d = A.shape
        v = (torch.ones(d) / np.sqrt(d)).to(device=DEVICE, dtype=DTYPE).view(d, 1)
        ev = self.eigenvalue(A, v)

        i = 0
        while True:
            i += 1
            Av = torch.mm(A, v)
            v_new = Av / torch.linalg.norm(Av.flatten())

            ev_new = self.eigenvalue(A, v_new)
            if torch.abs(ev - ev_new) < 0.001:
                break

            v = v_new
            ev = ev_new

        return ev_new, v_new.flatten()


def predict_degree_custom_model(adj_mat, y, reverse_nodes_mapping, g):
    zero_mat, const_mat, t_paths = get_fixed_mat(adj_mat, reverse_nodes_mapping, g)

    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    start_time = datetime.datetime.now()
    for t in range(200):
        y_pred = model(adj_mat, zero_mat, const_mat, t_paths)
        loss = criterion(y_pred, y)
        if t % 10 == 0 or t == 0:
            print(t, loss.item())
        # print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\nRun Time -  {datetime.datetime.now() - start_time} \n\nLearning Target - {y}')
    model_t = model.weights_t * t_paths
    model_r = model.weights_r * zero_mat + const_mat
    return model_t, model_r


def test_degree():
    # edge_lst = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    edge_lst = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]
    g = nx.DiGraph(edge_lst)
    nodes_mapping_reverse = {k: v for k, v in enumerate(list(g.nodes()))}
    adj_matrix = get_adj_mat(g, nodes_mapping_reverse)
    target_matrix = get_degree_in_out(adj_matrix)
    t_model, r_model = predict_degree_custom_model(adj_matrix, target_matrix, nodes_mapping_reverse, g)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(f"\n\nRBC Prediction - {rbc_pred}")


def test_betweenness():
    edge_lst = [(0, 1), (1, 2), (2, 3), (3, 4)]
    g = nx.DiGraph(edge_lst)
    nodes_mapping_reverse = {k: v for k, v in enumerate(list(g.nodes()))}
    adj_matrix = get_adj_mat(g, nodes_mapping_reverse)
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    betweenness_tensor = get_betweenness_tensor(g, nodes_mapping)
    t_model, r_model = predict_degree_custom_model(adj_matrix, betweenness_tensor, nodes_mapping_reverse, g)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(f'\n\nRBC Prediction - {rbc_pred}')


if __name__ == '__main__':
    test_degree()
    # test_betweenness()
