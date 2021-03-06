import random

from Utils.CommonStr import Centralities
import networkx as nx


class GraphGenerator:

    def __init__(self, centrality):
        self.centrality = centrality

    def generate_by_centrality(self):

        if self.centrality == Centralities.SPBC:
            return self.generate_spbc_graphs()
        else:
            return self.generate_rand_graphs()

    @staticmethod
    def generate_spbc_graphs():
        edge_lst_0 = [(0, 1), (0, 2), (1, 2), (2, 3)]
        edge_lst_1 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)]
        edge_lst_2 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5)]
        edge_lst_3 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
        edge_lst_4 = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 5)]
        edge_lst_5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
        edge_lst_6 = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4), (0, 4), (4, 5)]
        edge_lst_7 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4)]
        edge_lst_8 = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4), (5, 7),
                      (6, 7), (4, 8), (7, 8), (8, 9), (4, 9)]
        # all_edge_lst = [edge_lst_0, edge_lst_1, edge_lst_2, edge_lst_3, edge_lst_4, edge_lst_5, edge_lst_6, edge_lst_7,
        #                 edge_lst_8]
        all_edge_lst = [edge_lst_0, edge_lst_0, edge_lst_0, edge_lst_0, edge_lst_0]
        graphs = [nx.Graph(edges) for edges in all_edge_lst]
        return graphs

    @staticmethod
    def generate_rand_graphs(min_nodes, max_nodes, keep_rate=0.7):
        graphs = []
        for i in range(min_nodes, max_nodes):
            g = nx.complete_graph(i)
            edge_lst = list(nx.edges(g))
            rand_edges = random.sample(edge_lst, int(keep_rate * len(edge_lst)))
            h = nx.Graph()
            h.add_nodes_from(g)
            h.add_edges_from(rand_edges)
            largest_cc = max(nx.connected_components(h), key=len)
            largest_cc_graph = h.subgraph(largest_cc).copy()
            graphs.append(largest_cc_graph)
        return graphs

    @staticmethod
    def generate_n_nodes_graph(n, keep_rate=0.7):
        G = nx.complete_graph(n)
        edge_list = list(nx.edges(G))
        rand_edges = random.sample(edge_list, int(keep_rate * len(edge_list)))
        g = nx.Graph()
        g.add_nodes_from(G)
        g.add_edges_from(rand_edges)
        return [g]

    # def custom_graph(self):
    #     edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (1, 6), (4, 5), (5, 6)]
    #     g = nx.Graph(edge_list)
    #     return [g]

    def custom_graph(self):
        edge_list = [(0, 1), (1, 2), (2, 0), (2, 3)]
        g = nx.Graph(edge_list)
        return [g]
#
if __name__ == '__main__':
    GraphGenerator('bb').generate_rand_graphs(4, 1)
