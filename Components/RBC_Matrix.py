import networkx as nx
import numpy as np


def rbc(g, R, T):
    nodes_map = {k: v for v, k in enumerate(list(g.nodes()))}
    s_mapping = [nodes_map[node] for node in g.nodes()]
    t_mapping = s_mapping
    num_nodes = g.number_of_nodes()

    all_delta_arrays = [accumulate_delta(s, t, T, R(s, t), num_nodes) for s in s_mapping for t in t_mapping]
    rbc_arr = np.add.reduce(all_delta_arrays)
    return rbc_arr


def accumulate_delta(src, target, T, predecessor_prob_matrix, num_nodes):
    delta_arr = np.full(shape=(num_nodes, 1), fill_value=0, dtype=np.float)
    delta_arr[src] = T(src, target)

    predecessors = np.array([src])
    while predecessors.shape[0] > 0:
        delta_arr += np.matmul(predecessor_prob_matrix, delta_arr)
        new_predecessors = [np.nonzero(predecessor_prob_matrix[:, predecessors])[0]]
        predecessor_prob_matrix[:, predecessors] = 0
        predecessors = np.unique(np.concatenate(new_predecessors, axis=0))
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


def get_betweenness_policy_matrix(g, nodes_mapping):
    num_nodes = g.number_of_nodes()
    betweenness_matrix = np.full(shape=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0, dtype=float)
    for s in g.nodes():
        for t in g.nodes():
            edges_probabilities = get_edges_probabilities(g, s, t, nodes_mapping)
            betweenness_matrix[nodes_mapping[s]][nodes_mapping[t]] = edges_probabilities

    return betweenness_matrix


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
    # edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
    edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
             ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
             ('v2', 't'), ('v5', 't'), ('v3', 't')}
    graph = nx.DiGraph(edges)
    nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
    policy = get_betweenness_policy_matrix(graph, nodes_mapping)
    res = rbc(graph, lambda s, t: policy[s][t], lambda s, t: 100)
    a = 1
# <class 'tuple'>: (array([1116.66666667, 1133.33333333, 1172.22222222, 1000.        ,
#         700.        ,  883.33333333, 1179.62962963]), {'v1': 0, 'v5': 1, 'v2': 2, 'v3': 3, 's': 4, 'v4': 5, 't': 6})
