import networkx as nx
import numpy as np
import torch
import scipy as sp
from scipy.sparse import linalg


def rbc(g, R, T):
    nodes_map = {k: v for v, k in enumerate(list(g.nodes()))}
    s_mapping = [nodes_map[node] for node in g.nodes()]
    t_mapping = s_mapping
    all_delta_arrays = [accumulate_delta(s, t, R(s, t), T) for s in s_mapping for t in t_mapping]
    rbc_arr = torch.sum(torch.stack(all_delta_arrays), dim=0)
    return rbc_arr


def accumulate_delta(src, target, predecessor_prob_matrix, T):
    eigenvalues, eigenvectors = torch.eig(input=predecessor_prob_matrix, eigenvectors=True)
    eigenvector = get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, torch.tensor([[1.0, 0.0]]))
    eigenvector = compute_eigenvector_values(src, target, eigenvector, T)
    return eigenvector


def compute_eigenvector_values(src, target, eigenvector, T):
    x = 1 / float(eigenvector[src])  # the ratio between 1 to the current value of eigenvector[src]
    n_eigenvector = eigenvector * x * T(src, target)
    return n_eigenvector


def get_eigenvector_by_eigenvalue(eigenvalues, eigenvectors, eigenvalue):
    comparison_sum = (eigenvalues == eigenvalue).sum(dim=1)  # pair-wise compare and sum how many elements equal
    indicies = torch.nonzero(comparison_sum == eigenvalues.size(1))  # the equal indexes
    eigenvalue_idx = int(indicies[0][0])  # extract the index
    eigenvector = eigenvectors[:, eigenvalue_idx]  # extract the eigenvector col corresponding to eigenvalue index
    return eigenvector


def get_betweenness_policy_tensor(g, nodes_mapping):
    num_nodes = g.number_of_nodes()
    betweenness_tensor = torch.full(size=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0.0, dtype=float)
    for s in g.nodes():
        for t in g.nodes():
            edges_probabilities = get_edges_probabilities(g, s, t, nodes_mapping)
            betweenness_tensor[nodes_mapping[s]][nodes_mapping[t]] = edges_probabilities

    return betweenness_tensor


def get_edges_probabilities(g, src, target, nodes_mapping):
    matrix_prob = torch.full(size=(g.number_of_nodes(), g.number_of_nodes()), fill_value=0.0, dtype=float)
    matrix_prob[nodes_mapping[src], nodes_mapping[src]] = 1.0
    try:
        all_shortest_path = to_tuple_edge_paths(nx.all_shortest_paths(g, src, target))
    except Exception as e:  # shortest path not exist
        return matrix_prob
    edges = set([edge for path in all_shortest_path for edge in path])
    num_paths = len(all_shortest_path)
    for edge in edges:
        c_e = count_edge(all_shortest_path, edge)
        edge_probability = c_e / num_paths
        # matrix_prob[nodes_mapping[edge[1]]][nodes_mapping[edge[0]]] = edge_probability #TODO: return to old
        # matrix_prob[nodes_mapping[edge[0]]][nodes_mapping[edge[1]]] = edge_probability
        matrix_prob[nodes_mapping[edge[1]]][nodes_mapping[edge[0]]] = edge_probability

    return matrix_prob


def count_edge(all_shortest_path, edge):
    c = 0
    for path in all_shortest_path:
        if edge in path:
            c += 1
    return c


def to_tuple_edge_paths(all_shortest_path):
    all_edges_tuple_paths = []
    for path in all_shortest_path:
        tuple_path = []
        for i in range(0, len(path) - 1):
            tuple_path.append((path[i], path[i + 1]))
        all_edges_tuple_paths.append(tuple_path)

    return all_edges_tuple_paths


if __name__ == '__main__':
    edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
    # edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
    #          ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
    #          ('v2', 't'), ('v5', 't'), ('v3', 't')}
    graph = nx.DiGraph(edges)
    nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
    policy = get_betweenness_policy_tensor(graph, nodes_mapping)
    res = rbc(graph, lambda s, t: policy[s][t], lambda s, t: 100)
