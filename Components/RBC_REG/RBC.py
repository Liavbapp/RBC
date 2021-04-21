import networkx as nx
import torch
import Components.RBC_REG.Policy as Policy
from Components.RBC_ML.PowerIteration import PowerIteration
from Utils.CommonStr import EigenvectorMethod


class RBC:
    def __init__(self, eigenvector_method, pi_max_error, device, dtype):
        self.eigenvector_method = eigenvector_method
        self.device = device
        self.dtype = dtype
        self.pi_handler = PowerIteration(device=self.device, dtype=self.dtype, max_error=pi_max_error)

    def compute_rbcs(self, Gs, Rs, Ts):
        return torch.stack([self.compute_rbc(g, R, T) for g, R, T in zip(Gs, Rs, Ts)])


    def compute_rbc(self, g, R, T):
        nodes_map = {k: v for v, k in enumerate(list(g.nodes()))}
        s_mapping = [nodes_map[node] for node in g.nodes()]
        t_mapping = s_mapping
        all_delta_arrays = [self.accumulate_delta(s, R[s, t], T[s, t], t) for s in s_mapping for t in t_mapping]
        rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
        return rbc_arr

    def accumulate_delta(self, src, predecessor_prob_matrix, T_val, t):
        if self.eigenvector_method == EigenvectorMethod.torch_eig:
            eigenvalues, eigenvectors = torch.eig(input=predecessor_prob_matrix, eigenvectors=True)
            # eigenvector = get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, torch.tensor([[1.0, 0.0]]))  # todo: find out which impl is right this or below
            eigenvector = eigenvectors[:, torch.argmax(eigenvalues.t()[0])]
        elif self.eigenvector_method == EigenvectorMethod.power_iteration:
            eigenvalues, eigenvector = self.pi_handler.power_iteration(A=predecessor_prob_matrix)
        else:
            raise NotImplementedError
        normalized_eigenvector = self.normalize_eigenvector(src, eigenvector, T_val)

        return normalized_eigenvector

    def normalize_eigenvector(self, src, eigenvector, T_val):
        x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
        n_eigenvector = eigenvector * x * T_val

        return n_eigenvector

    def get_eigenvector_by_eigenvalue(self, eigenvalues, eigenvectors, eigenvalue):
        comparison_sum = (eigenvalues == eigenvalue).sum(dim=1)  # pair-wise compare and sum how many elements equal
        indicies = torch.nonzero(comparison_sum == eigenvalues.size(1))  # the equal indexes
        eigenvalue_idx = int(indicies[0][0])  # extract the index
        eigenvector = eigenvectors[:, eigenvalue_idx]  # extract the eigenvector col corresponding to eigenvalue index
        return eigenvector


if __name__ == '__main__':
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    betweenness_policy = Policy.BetweennessPolicy()
    graph = nx.Graph(edges)
    nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
    nodes_mapping_reverse = {k: v for k, v in enumerate(list(graph.nodes()))}
    R = betweenness_policy.get_policy_tensor(graph, nodes_mapping)
    T = betweenness_policy.get_t_tensor(graph, nodes_mapping_reverse)
    res = RBC(eigenvector_method=EigenvectorMethod.power_iteration, pi_max_error=0.0001,
              device=torch.device('cpu'), dtype=torch.float).compute_rbc(graph, R, T)

    print(f'networkx result: {nx.betweenness_centrality(graph, endpoints=True, normalized=False).values()}, actual result: {res}')
