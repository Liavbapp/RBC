import datetime
import torch
import networkx as nx
from Components import Policy
import Components.RBC as RBC
import numpy as np

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
            torch.rand(self.num_nodes, self.num_nodes, requires_grad=True, device=device, dtype=dtype))
        self.weights_r = torch.nn.Parameter(torch.rand(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                                                       requires_grad=True, device=device, dtype=dtype))

        self.self_eigenvalue = torch.tensor([[1.0, 0.0]], device=DEVICE, dtype=DTYPE)

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

    def power_iteration(self, A, num_iter=10):
        n, d = A.shape
        v = (torch.ones(d) / np.sqrt(d)).to(device=DEVICE, dtype=DTYPE).view(d, 1)
        ev = self.eigenvalue(A, v)

        i = 0
        while i < num_iter:
            i += 1
            Av = torch.mm(A, v)
            v_new = Av / torch.linalg.norm(Av.flatten())

            ev_new = self.eigenvalue(A, v_new)
            # if torch.abs(ev - ev_new) < 0.01:
            #     break

            v = v_new
            ev = ev_new

        return ev_new, v_new.flatten()


def predict_degree_custom_model(adj_mat, y):
    zero_mat, const_mat = get_fixed_mat(adj_mat)

    model = DegreePrediction(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = datetime.datetime.now()
    for t in range(800):
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
    adj_matrix_t = adj_mat.t()
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1
            zeros_mat[s, t] = adj_matrix_t
            zeros_mat[s, t, s] = 0  # we want to zeros all edges that go into the source node

    return zeros_mat, constant_mat


def test_degree():
    edge_lst = [(0, 1), (1, 2), (2, 0), (2, 3)]
    g = nx.DiGraph(edge_lst)
    # g = nx.watts_strogatz_graph(n=10, k=3, p=0.5)
    adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    target_matrix = torch.tensor(list(map(float, dict(nx.degree(g)).values())), device=DEVICE, dtype=DTYPE)
    # g = g.to_directed()
    t_model, r_model = predict_degree_custom_model(adj_matrix, target_matrix)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)


def test_betweenness():
    edge_lst = [(0, 1), (1, 2), (2, 0), (2, 3)]
    g = nx.DiGraph(edge_lst)
    adj_matrix = torch.from_numpy(nx.to_numpy_matrix(g)).to(dtype=DTYPE, device=DEVICE)
    betweenness_tensor = torch.tensor(list(nx.betweenness_centrality(g).values()), dtype=DTYPE, device=DEVICE)
    t_model, r_model = predict_degree_custom_model(adj_matrix, betweenness_tensor)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC.rbc(g, r_model, t_model)
    print(rbc_pred)


if __name__ == '__main__':
    test_betweenness()
