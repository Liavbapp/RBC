import torch
from Components.RBC_ML.PowerIteration import power_iteration

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


class RbcNetwork(torch.nn.Module):
    def __init__(self, adj_mat):
        super().__init__()
        self.num_nodes = adj_mat.size()[0]
        self.weights_t = torch.nn.Parameter(
            torch.rand(self.num_nodes, self.num_nodes, requires_grad=True, device=DEVICE, dtype=DTYPE))
        self.weights_r = torch.nn.Parameter(torch.rand(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                                                       requires_grad=True, device=DEVICE, dtype=DTYPE))

    def forward(self, x, r_zeros, r_const, max_error):
        weights_t_fixed = self.weights_t
        weights_r_comb = torch.sigmoid(torch.mul(self.weights_r, r_zeros) + r_const)
        all_delta_arrays = [self.accumulate_delta(s, weights_r_comb[s, t], weights_t_fixed[s, t], max_error) for s in
                            range(0, len(x)) for t in range(0, len(x))]
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
        return rbc_arr

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val, max_error):
        new_eigenvalue, eigenvector = power_iteration(A=predecessor_prob_matrix, max_error=max_error)
        eigenvector2 = eigenvector
        eigenvector = self.compute_eigenvector_values(src, eigenvector2, T_val)
        return eigenvector

    def compute_eigenvector_values(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector
