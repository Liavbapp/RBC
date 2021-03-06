import abc
import networkx as nx
import torch


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_policy_tensor(self, graph, nodes_mapping):
        pass


class DegreePolicy(Policy):

    def get_t_tensor(self, graph: nx.Graph):
        nodes_mapping = {k: v for v, k in enumerate(list(graph.nodes()))}
        t_tensor = torch.full(size=(graph.number_of_nodes(), graph.number_of_nodes()), fill_value=0.0)
        for s in graph.nodes():
            for t in graph.nodes():
                if graph.has_edge(s, t):
                    t_tensor[nodes_mapping[s], nodes_mapping[t]] = 0.5
        return t_tensor

    def get_policy_tensor(self, graph: nx.Graph, nodes_mapping):
        num_nodes = graph.number_of_nodes()
        deg_tensor = torch.full(size=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0.0)
        for s in graph.nodes():
            for t in graph.nodes():
                st_deg_matrix = torch.full(size=(num_nodes, num_nodes), fill_value=0.0)
                if graph.has_edge(s, t):
                    st_deg_matrix[nodes_mapping[t], nodes_mapping[s]] = 1.0
                st_deg_matrix[nodes_mapping[s], nodes_mapping[s]] = 1.0
                deg_tensor[nodes_mapping[s], nodes_mapping[t]] = st_deg_matrix
        return deg_tensor


class BetweennessPolicy(Policy):

    def get_t_tensor(self, graph, nodes_mapping_reverse):

        num_nodes = len(nodes_mapping_reverse)
        traffic_mat = torch.full(size=(num_nodes, num_nodes), fill_value=0.5)
        for s in range(0, num_nodes):
            for t in range(0, num_nodes):
                if s == t:
                    traffic_mat[s, t] = 0.0
                else:
                    if nx.has_path(graph, nodes_mapping_reverse[s], nodes_mapping_reverse[t]):
                        traffic_mat[s, t] = 0.5
                    else:
                        traffic_mat[s, t] = 0.0

        return traffic_mat

    def get_policy_tensor(self, graph, nodes_mapping):
        num_nodes = graph.number_of_nodes()
        betweenness_tensor = torch.full(size=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0.0)
        for s in graph.nodes():
            for t in graph.nodes():
                edges_probabilities = self.get_edges_probabilities(graph, s, t, nodes_mapping)
                betweenness_tensor[nodes_mapping[s]][nodes_mapping[t]] = edges_probabilities

        return betweenness_tensor


    def get_edges_probabilities(self, g, src, target, nodes_mapping):
        matrix_prob = torch.full(size=(g.number_of_nodes(), g.number_of_nodes()), fill_value=0.0)
        matrix_prob[nodes_mapping[src], nodes_mapping[src]] = 1.000
        try:
            all_shortest_path = self.to_tuple_edge_paths(nx.all_shortest_paths(g, src, target))
        except Exception as e:  # shortest path not exist
            return matrix_prob
        edges = set([edge for path in all_shortest_path for edge in path])
        edges_src_count = {}
        for edge in edges:
            if edge[0] not in edges_src_count:
                edges_src_count[edge[0]] = 1
            else:
                edges_src_count[edge[0]] += 1
        for edge in edges:
            edge_prob = 1/edges_src_count[edge[0]]
            matrix_prob[nodes_mapping[edge[1]]][nodes_mapping[edge[0]]] = edge_prob
        # num_paths = len(all_shortest_path)
        # for edge in edges:
        #     c_e = self.count_edge(all_shortest_path, edge)
        #     edge_probability = c_e / num_paths
        #     matrix_prob[nodes_mapping[edge[1]]][nodes_mapping[edge[0]]] = edge_probability

        return matrix_prob

    def to_tuple_edge_paths(self, all_shortest_path):
        all_edges_tuple_paths = []
        for path in all_shortest_path:
            tuple_path = []
            for i in range(0, len(path) - 1):
                tuple_path.append((path[i], path[i + 1]))
            all_edges_tuple_paths.append(tuple_path)

        return all_edges_tuple_paths

    def count_edge(self, all_shortest_path, edge):
        c = 0
        for path in all_shortest_path:
            if edge in path:
                c += 1
        return c
