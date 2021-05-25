import os
import random
import numpy as np
from Utils.CommonStr import Centralities
import networkx as nx
from karateclub.dataset import GraphReader
import errno, os, stat, shutil

reader = GraphReader("twitch")
import shutil


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

    def small_world_gxef_graph(self, path):
        graphs = [f.path for f in os.scandir(path)]
        graphs = [nx.read_gexf(g) for g in graphs]
        return graphs

    def small_world_graphs(self, num_nodes_lst, k_lst, p_lst):
        lst_graphs = []
        for i in range(0, 6):
            for num_nodes in num_nodes_lst:
                for k in k_lst:
                    for p in p_lst:
                        g = nx.newman_watts_strogatz_graph(num_nodes, k, p, seed=None)
                        if nx.is_connected(g):
                            if set(g.edges) not in [set(edges) for edges in lst_graphs]:
                                lst_graphs.append(g)
        return lst_graphs

    def same_num_nodes_same_num_edges_diffrent_graphs(self, num_nodes, num_edges):
        lst_graphs = []
        while len(lst_graphs) < 100:
            g = nx.Graph()
            g.add_nodes_from(range(0, num_nodes))
            while not nx.is_connected(g) or len(g.edges) < num_edges:
                u, v = random.sample(range(0, num_nodes), 2)
                if not g.has_edge(u, v):
                    g.add_edge(u, v)
                if len(g.edges) == num_edges:
                    break
            if nx.is_connected(g):
                if set(g.edges) not in [set(edges) for edges in lst_graphs]:
                    lst_graphs.append(g)
        return lst_graphs

    def same_num_nodes_different_num_edges_graphs(self, num_nodes):
        lst_graphs = []
        edges_range = range(num_nodes + 5, num_nodes + 25)
        for num_edges in edges_range:
            g = nx.Graph()
            g.add_nodes_from(range(0, num_nodes))
            while not nx.is_connected(g) or len(g.edges) < num_edges:
                u, v = random.sample(range(0, num_nodes), 2)
                if not g.has_edge(u, v):
                    g.add_edge(u, v)
                if len(g.edges) == num_edges:
                    break
            if nx.is_connected(g):
                if set(g.edges) not in [set(edges) for edges in lst_graphs]:
                    lst_graphs.append(g)
        return lst_graphs

    def custom_graph(self):
        pass

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

        return lst

    def split_graphs_train_val_test(self, path, root_folder):
        all_sub_dir = []
        for dirpath, dirnames, filenames in os.walk(path):
            if not dirnames:
                all_sub_dir.append(dirpath)
        # immediate_sub = [f.path for f in os.scandir(path) if f.is_dir()]
        # for immediate_sub_dir in immediate_sub:
        #     all_sub_dir += [f.path for f in os.scandir(immediate_sub_dir)]

        num_sub_dir = len(all_sub_dir)
        indices = list(range(0, num_sub_dir))
        random.shuffle(indices)
        n_train, n_val, n_test = int(0.7 * num_sub_dir), int(0.1 * num_sub_dir), int(0.2 * num_sub_dir)
        train_indices, val_indices, test_indices = indices[:n_train], indices[n_train: n_train + n_val], indices[
                                                                                                         n_train + n_val:]

        for i, train_index in enumerate(train_indices):
            old_path = all_sub_dir[train_index]
            shutil.move(old_path, root_folder + f'\\train\\{str(i)}')

        for i, val_index in enumerate(val_indices):
            old_path = all_sub_dir[val_index]
            shutil.move(old_path, root_folder + f'\\validation\\{str(i + n_train)}')

        for i, test_index in enumerate(test_indices):
            old_path = all_sub_dir[test_index]
            shutil.move(old_path, root_folder + f'\\test\\{str(i + n_train + n_val)}')

        #
        # for i, path in enumerate(all_sub_dir):
        #     new_path = path[:-1] + str(i)
        #     os.rename(path, new_path)
        #     shutil.move(new_path, root_folder + f'\\{str(i)}')


if __name__ == '__main__':
    # GraphGenerator('bb').generate_rand_graphs(4, 1)
    # GraphGenerator('bb').custom_graph()
    # GraphGenerator('bb').read_gxef(path=r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\NCA-GE\Graphs\Train')
    GraphGenerator('bb').split_graphs_train_val_test(
        r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_5\Data\SPBC',
        r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_5\Data\SPBC')
    # for res in GraphGenerator('bb').same_num_nodes_different_num_edges_graphs(10):
    #     nx.draw(res, with_labels=True)
    #     plt.show()
    #     # print(lst)
