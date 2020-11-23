import networkx as nx
import numpy as np


def rbc(g: nx.DiGraph, R, T):
    rbc_arr = np.full(shape=g.number_of_nodes(), fill_value=0, dtype=np.float)
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    for s in g.nodes():
        for t in g.nodes():
            rbc_source_target(g, R, T, rbc_arr, s, t, nodes_mapping)

    return rbc_arr, nodes_mapping


def rbc_source_target(g, R, T, rbc_arr, s, t, nodes_mapping):
    dag: nx.DiGraph = get_dag(g, R, s, t)
    sorted_nodes = list(nx.topological_sort(dag))
    delta_arr = init_delta_arr(dag.number_of_nodes(), init_val=0, T=T, source=s, target=t, nodes_mapping=nodes_mapping)
    accumulate(dag, R, delta_arr, sorted_nodes, nodes_mapping, rbc_arr, s, t)


def get_dag(g, R, source, target):
    edges = relevant_edges(g, R, source, target)
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(g)

    return dag


def relevant_edges(g: nx.DiGraph, R, source_node, target_node):
    edges = set()
    for edge in g.edges():
        u = edge[0]
        v = edge[1]
        if R(source_node, u, v, target_node) > 0:
            edges.add(edge)

    return edges


def init_delta_arr(num_nodes, init_val, T, source, target, nodes_mapping):
    delta_arr = np.full(shape=num_nodes, fill_value=init_val, dtype=np.float)
    delta_arr[nodes_mapping[source]] = T(source, target)

    return delta_arr


def accumulate(dag, R, delta_arr, sorted_nodes, nodes_mapping, rbc_arr, source, target):
    delta_arr = update_delta_arr(dag, R, delta_arr, sorted_nodes, nodes_mapping, source, target)
    accumulate_to_rbc_arr(dag, rbc_arr, delta_arr, nodes_mapping)


def update_delta_arr(dag, R, delta_arr, sorted_nodes, nodes_mapping, source, target):
    for node_v in sorted_nodes:
        node_v_idx = nodes_mapping[node_v]
        successors = list(dag.neighbors(node_v))
        for successor_j in successors:
            successor_j_idx = nodes_mapping[successor_j]
            delta_arr[successor_j_idx] += delta_arr[node_v_idx] * R(source, node_v, successor_j, target)

    return delta_arr


def accumulate_to_rbc_arr(dag, rbc_arr, delta_arr, nodes_mapping):
    for node_v in dag.nodes():
        node_v_idx = nodes_mapping[node_v]
        rbc_arr[node_v_idx] += delta_arr[node_v_idx]


def get_edges_probabilities(g, src, target):
    edges_prob = {}
    try:
        all_shortest_path = to_tuple_edge_paths(nx.all_shortest_paths(g, src, target))
    except Exception as e:  # shortest path not exist
        return {}
    edges = set([edge for path in all_shortest_path for edge in path])
    num_paths = len(all_shortest_path)
    for edge in edges:
        c_e = count_edge(all_shortest_path, edge)
        edge_probability = c_e / num_paths
        edges_prob.update({edge: edge_probability})

    return edges_prob




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


def get_betweenness_policy_dict(g):
    betweenness_dict = {}
    i = 0
    num_nodes = len(g.nodes())
    for s in g.nodes():
        i += 1
        j = 0
        for t in g.nodes():
            j += 1
            print(f' source node {i} out of {num_nodes}, target node {j} out of {num_nodes}')
            edges_probabilities = get_edges_probabilities(g, s, t)
            betweenness_dict.update({(s, t): edges_probabilities})

    return betweenness_dict


# def rbc(g: nx.DiGraph, R, T):
#     rbc_arr = np.full(shape=g.number_of_nodes(), fill_value=0, dtype=np.int)
#     nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
#     i = 0
#     num_nodes = len(g.nodes())
#     for s in g.nodes():
#         i += 1
#         j = 0
#         for t in g.nodes():
#             j += 1
#             print(f' source node {i} out of {num_nodes}, target node {j} out of {num_nodes}')
#             rbc_source_target(g, R, T, rbc_arr, s, t, nodes_mapping)
#     return rbc_arr, nodes_mapping


# def save_obj(obj, dir_path, f_name):
#     with open(dir_path + "\\" + f_name + ".pkl", 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#
#
# def load_obj(dir_path, f_name):
#     with open(dir_path + "\\" + f_name + ".pkl", 'rb') as f:
#         return pickle.load(f)


def accumulate_matrix(delta_arr, src, target, R, nodes_mapping, g):
    predecessor_prob_matrix = R(src, target)
    predecessors = {src}
    while len(predecessors) > 0:
        delta_arr += np.matmul(predecessor_prob_matrix, delta_arr)
        new_predecessors = set()
        for old_predecessor in predecessors:
            new_predecessors = new_predecessors.union(set(g.neighbors(old_predecessor)))
            predecessor_prob_matrix[:, nodes_mapping[old_predecessor]] = 0
        predecessors = new_predecessors
    return delta_arr


if __name__ == '__main__':
    #      g = nx.read_edgelist(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\test_graphs\CSV\ColiNet-1.1\grass_web\grass_web.pairs')
    #     r = get_betweenness_policy_dict(g)
    #     r = load_obj(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\nisuy', 'betweennes')
    # edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
    #          ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
    #          ('v2', 't'), ('v5', 't'), ('v3', 't')}
    edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
    g = nx.DiGraph(edges)
    policy = get_betweenness_policy_dict(g)
    res = rbc(g, lambda s, u, v, t: policy[(s, t)][(u, v)] if (u, v) in policy[(s, t)] else 0, lambda s, t: 100)
    a = 1



