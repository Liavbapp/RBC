import torch
from Components.RBC_ML.PowerIteration import PowerIteration
from Utils.CommonStr import EigenvectorMethod


class RbcNetwork(torch.nn.Module):
    def __init__(self, num_nodes, use_sigmoid, cosnider_traffic_paths, pi_max_err, eigenvector_method, device, dtype,
                 fixed_t, fixed_R):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.num_nodes = num_nodes
        self.zero_traffic = cosnider_traffic_paths
        if fixed_t is None:
            self.weights_t = torch.nn.Parameter(
                torch.rand(self.num_nodes, self.num_nodes, requires_grad=True, device=device, dtype=dtype))
        else:
            self.weights_t = fixed_t
        if fixed_R is None:
            self.weights_r = torch.nn.Parameter(
                torch.rand(self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes,
                           requires_grad=True, device=device, dtype=dtype))
        else:
            self.weights_r = fixed_R
        self.pi_handler = PowerIteration(device=device, dtype=dtype, max_error=pi_max_err)
        self.eigenvector_method = eigenvector_method

    def forward(self, r_zeros, r_const, traffic_paths):
        weights_t_fixed = torch.mul(self.weights_t, traffic_paths) if self.zero_traffic else self.weights_t
        weights_r_comb = (torch.mul(torch.sigmoid(self.weights_r), r_zeros) + r_const) if self.use_sigmoid else \
            torch.mul(self.weights_r, r_zeros) + r_const

        all_delta_arrays = [self.accumulate_delta(s, weights_r_comb[s, t], weights_t_fixed[s, t]) for s in
                            range(0, self.num_nodes) for t in range(0, self.num_nodes)]
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
        return rbc_arr

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val):
        if self.eigenvector_method == EigenvectorMethod.power_iteration:
            new_eigenvalue, eigenvector = self.pi_handler.power_iteration(A=predecessor_prob_matrix)
        else:
            raise NotImplementedError
        normalized_eigenvector = self.noramlize_eiginevector(src, eigenvector, T_val)
        return normalized_eigenvector

    def noramlize_eiginevector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x
        n_eigenvector = n_eigenvector * T_val

        return n_eigenvector
