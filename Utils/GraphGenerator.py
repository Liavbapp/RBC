import random
import matplotlib.pyplot as plt
from Utils.CommonStr import Centralities
import networkx as nx
from karateclub.dataset import GraphReader
import numpy as np
reader = GraphReader("twitch")
import itertools

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
        edges = [(0, 9), (0, 8), (9, 2), (8, 2), (9, 10), (4, 10), (1, 10), (1, 3), (3, 5), (10, 5), (2, 5), (2, 10), (6, 5), (2, 6), (2, 7)]
        g = nx.Graph()
        g.add_nodes_from(range(0, 11))
        g.add_edges_from(edges)
        return [g]

        # g = nx.Graph()
        # g.add_nodes_from(list(range(13)))
        # graphs = []
        # prev_edges = []
        # for i in range(100):
        #     g = nx.complete_graph(13)
        #     edge_lst = list(nx.edges(g))
        #     rand_edges = random.sample(edge_lst, int(random.uniform(0.5, 1) * len(edge_lst)))
        #     if rand_edges not in prev_edges:
        #         prev_edges.append(rand_edges)
        #         h = nx.Graph()
        #         h.add_nodes_from(g)
        #         h.add_edges_from(rand_edges)
        #         # largest_cc = max(nx.connected_components(h), key=len)
        #         # largest_cc_graph = h.subgraph(largest_cc).copy()
        #         graphs.append(h)
        # return graphs
        graphs = []
        # g1 = nx.complete_graph(5)
        # g2 = nx.complete_graph(5)
        # g3 = nx.Graph([(0, 1), (1, 2), (2, 3)])
        # h = nx.disjoint_union_all([g1, g3, g2])
        # h.add_edge(0, 5)
        # h.add_edge(8, 9)
        # nx.draw(h, with_labels=True)
        # return [h]


        # prev_edges = []
        # for i in range(10, 30):
        #     g = nx.complete_graph(i)
        #     edge_lst = list(nx.edges(g))
        #     rand_edges = random.sample(edge_lst, int(random.uniform(0.65, 1) * len(edge_lst)))
        #     h = nx.Graph()
        #     h.add_nodes_from(g)
        #     h.add_edges_from(rand_edges)
        #     largest_cc = max(nx.connected_components(h), key=len)
        #     largest_cc_graph = h.subgraph(largest_cc).copy()
        #     graphs.append(largest_cc_graph)
        # return graphs
            # if rand_edges not in prev_edges:
            #     prev_edges.append(rand_edges)
            #     h = nx.Graph()
            #     h.add_nodes_from(g)
            #     h.add_edges_from(rand_edges)
            #     # largest_cc = max(nx.connected_components(h), key=len)
            #     # largest_cc_graph = h.subgraph(largest_cc).copy()
            #     graphs.append(h)
        # return graphs
        # edge_list = [(0, 1), (1, 2), (2, 0), (2, 3)]
        # g = nx.Graph(edge_list)
        # return [g]
        # c = nx.complete_graph(4)
        # d = nx.complete_graph(4)
        # a = nx.Graph([(0, 1), (1, 2)])
        # h = nx.disjoint_union_all([a, c, d])
        # h.add_edge(0, 3)
        # h.add_edge(2, 10)
        # return [h]
        #
        # for i
        # g0 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
        # g1 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        # g2 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)])
        # g3 = nx.complete_graph(5)
        #
        # g4 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 2), (4, 1)])
        # g5 = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (1, 4), (3, 0)])
        # g6 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)])
        # g7 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])
        # g8 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (1, 4)])
        # g9 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)])
        # g10 = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])
        # g11 = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (2, 1), (2, 0), (2, 4)])
        # g12 = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (3, 1), (3, 2), (3, 4)])
        # g13 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 2), (4, 1), (2, 3)])
        # g14 = nx.Graph()
        # g14.add_nodes_from([0, 1, 2, 3, 4])
        # g14.add_edges_from([(0, 1), (0, 2), (2, 4)])
        # g15 = nx.Graph()
        # g15.add_nodes_from([0, 1, 2, 3, 4])
        # g15.add_edges_from([(1, 2), (2, 4)])
        # random.sample(list(set(itertools.permutations(lst))), 0.05)
        #
        # return [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15]


    def graphs_for_embeddings_show(self):
        lst = []
        c = nx.complete_graph(40)
        d = nx.complete_graph(40)
        a = nx.Graph([(0, 1), (1, 2)])
        b = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])
        h = nx.disjoint_union_all([a, b, c, d])
        h.add_edge(11, 87)
        h.add_edge(3, 18)
        h.add_edge(0, 74)
        h.add_edge(2, 19)
        lst.append(h)
        # path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\15'
        # lst.append(nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')))
        # nx.draw_networkx(h, with_labels=True, node_size=500)
        # plt.show()
        #
        # edge_list = [(0, 1), (1, 2), (2, 0), (2, 3)]
        # g = nx.Graph(edge_list)
        # lst.append(g)
        # lst.append(nx.Graph([(0, 1), (1, 2), (1, 3), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 0)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]))
        # lst.append(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 6)]))
        # lst.append(nx.Graph(nx.complete_graph(5)))
        # lst.append(nx.Graph(nx.complete_graph(10)))
        # lst.append(nx.complete_graph(15))
        # lst.append(nx.complete_graph(20))
        # lst.append(nx.complete_graph(25))
        # lst.append(nx.complete_graph(30))
        # lst.append(nx.complete_graph(35))
        # lst.append(nx.complete_graph(40))
        # lst.append(nx.complete_graph(45))
        # lst.append(nx.complete_graph(50))
        return lst


if __name__ == '__main__':
    # GraphGenerator('bb').generate_rand_graphs(4, 1)
    GraphGenerator('bb').custom_graph()