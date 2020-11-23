import networkx as nx
import numpy as np


def rbc(g: nx.DiGraph, R, T):
    rbc_arr = np.full(shape=(g.number_of_nodes(), 1), fill_value=0, dtype=np.float)
    nodes_map = {k: v for v, k in enumerate(list(g.nodes()))}
    for s in g.nodes():
        for t in g.nodes():
            rbc_source_target(g.number_of_nodes(), R, T, rbc_arr, nodes_map[s], nodes_map[t])
    return rbc_arr, nodes_map


def rbc_source_target(num_nodes, R, T, rbc_arr, s, t):
    delta_arr = init_delta_arr(num_nodes, T=T, source=s, target=t)
    updated_delta = accumulate_delta(delta_arr, s, t, R)
    rbc_arr += updated_delta


def init_delta_arr(num_nodes, T, source, target):
    delta_arr = np.full(shape=(num_nodes, 1), fill_value=0, dtype=np.float)
    delta_arr[source] = T(source, target)

    return delta_arr


def accumulate_delta(delta_arr, src, target, R):
    predecessor_prob_matrix = R(src, target)
    predecessors = {src}
    while len(predecessors) > 0:
        delta_arr += np.matmul(predecessor_prob_matrix, delta_arr)
        new_predecessors = set()
        for old_predecessor in predecessors:
            new_predecessors = new_predecessors.union(set(np.nonzero(predecessor_prob_matrix[:, ])[0]))
            predecessor_prob_matrix[:, old_predecessor] = 0
        predecessors = new_predecessors
    return delta_arr


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


def get_betweenness_policy_dict(g, nodes_mapping):
    betweenness_dict = {}
    for s in g.nodes():
        for t in g.nodes():
            edges_probabilities = get_edges_probabilities(g, s, t, nodes_mapping)
            betweenness_dict.update({(nodes_mapping[s], nodes_mapping[t]): edges_probabilities})

    return betweenness_dict


def get_edges_probabilities(g, src, target, nodes_mapping):
    matrix_prob = np.full(shape=(g.number_of_nodes(), g.number_of_nodes()), fill_value=0, dtype=float)
    try:
        all_shortest_path = to_tuple_edge_paths(nx.all_shortest_paths(g, src, target))
    except Exception as e:  # shortest path not exist
        return matrix_prob
    edges = set([edge for path in all_shortest_path for edge in path])
    num_paths = len(all_shortest_path)
    for edge in edges:
        c_e = count_edge(all_shortest_path, edge)
        edge_probability = c_e / num_paths
        matrix_prob[nodes_mapping[edge[1]]][nodes_mapping[edge[0]]] = edge_probability

    return matrix_prob


if __name__ == '__main__':
    edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
    g = nx.DiGraph(edges)
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    policy = get_betweenness_policy_dict(g, nodes_mapping)
    source = 'v1'
    dest = 'v4'
    res = rbc(g, lambda s, t: policy[(s, t)], lambda s, t: 100)
