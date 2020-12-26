import networkx as nx
import torch
import Components.Policy as Policy

DEVICE = 'cuda:0'
DTYPE = 'float'

def rbc(g, R, T):
    nodes_map = {k: v for v, k in enumerate(list(g.nodes()))}
    s_mapping = [nodes_map[node] for node in g.nodes()]
    t_mapping = s_mapping
    all_delta_arrays = [accumulate_delta(s, R[s, t], T[s, t]) for s in s_mapping for t in t_mapping]
    rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
    return rbc_arr


def accumulate_delta(src, predecessor_prob_matrix, T_val):
    eigenvalues, eigenvectors = torch.eig(input=predecessor_prob_matrix, eigenvectors=True)
    eigenvector = get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, torch.tensor([[1.0, 0.0]]))
    eigenvector = compute_eigenvector_values(src, eigenvector, T_val)
    return eigenvector


def compute_eigenvector_values(src, eigenvector, T_val):
    x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
    n_eigenvector = eigenvector * x * T_val
    return n_eigenvector


def get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, eigenvalue):
    comparison_sum = (eigenvalues == eigenvalue).sum(dim=1)  # pair-wise compare and sum how many elements equal
    indicies = torch.nonzero(comparison_sum == eigenvalues.size(1))  # the equal indexes
    eigenvalue_idx = int(indicies[0][0])  # extract the index
    eigenvector = eigenvectors[:, eigenvalue_idx]  # extract the eigenvector col corresponding to eigenvalue index
    return eigenvector


# def compute_delta_ratios(predecessor_prob_matrix):
#     eigenvalues, eigenvectors = torch.eig(input=predecessor_prob_matrix, eigenvectors=True)
#     eigenvector = get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, torch.tensor([[1.0, 0.0]]))
#     return eigenvector
#
#


if __name__ == '__main__':
    # edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
    # # edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
    # #          ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
    # #          ('v2', 't'), ('v5', 't'), ('v3', 't')}
    # betweenness_policy = Policy.BetweennessPolicy()
    # graph = nx.DiGraph(edges)
    # nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
    # R = betweenness_policy.get_policy_tensor(graph, nodes_mapping)
    # T = betweenness_policy.get_t_tensor(graph)
    # res = rbc(graph, R, T)
    # edges = {('v1', 'v2'), ('v1', 'v3'), ('v5', 'v8'), ('v1', 'v5')}
    # graph = nx.Graph(edges)
    # graph.add_node('v10')
    deg_policy = Policy.DegreePolicy()
    edges_g1 = [('v0', 'v1')]
    # edges_g1 = {('v0', 'v1'), ('v0', 'v2'), ('v1', 'v2'), ('v3', 'v2'), ('v1', 'v3'), ('v0', 'v4'), ('v1', 'v5'),
    #             ('v6', 'v7'), ('v8', 'v3'), ('v5', 'v9'), ('v10', 'v8')}
    graph = nx.Graph(edges_g1).to_directed()
    nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
    R = deg_policy.get_policy_tensor(graph, nodes_mapping)
    T = deg_policy.get_t_tensor(graph)
    res = rbc(graph, R, T)
    # print(nodes_mapping)
    print(res)
