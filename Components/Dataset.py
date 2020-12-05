import networkx as nx
import torch
import random
from torch_geometric.data import InMemoryDataset, Data

from Components import Policy


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset\graph_datset.dataset']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.get_data_list()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        pass

    def get_data_list(self):
        degree_policy = Policy.DegreePolicy()
        graphs = self.get_graphs()
        data_list = []
        for g in graphs:
            node_map = {k: v for v, k in enumerate(list(g.nodes()))}
            R, T = self.get_policies(g, node_map, degree_policy)
            data_list.append(self.get_data_object(R, T, g, node_map))
        return data_list

    @staticmethod
    def get_policies(g, nodes_map, policy):
        r = policy.get_policy_tensor(g, nodes_map)
        t = policy.get_t_tensor(g)
        return r, t

    def get_data_object(self, R, T, g, nodes_map):
        x = self.get_features_vec(R, T)
        y = self.get_labels_vec(g, nodes_map)
        train_set, validation_set, test_set = self.get_data_division(len(nodes_map))
        edge_index = self.get_edge_index(g, nodes_map)
        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_set, test_mask=test_set, validation_mask=validation_set)
        return data

    @staticmethod
    def get_features_vec(r, t):
        r_flatten = torch.flatten(r, start_dim=1)
        vec = torch.cat((r_flatten, t), dim=1)
        return vec

    @staticmethod
    def get_labels_vec(g, nodes_map):
        degrees = [0] * len(nodes_map)
        for node, degree in g.out_degree():
            degrees[nodes_map[node]] = degree
        # degrees = [degree[1] for degree in g.out_degree()]
        return torch.tensor(degrees)

    @staticmethod
    def get_edge_index(g: nx.Graph, nodes_map):
        lst_src = []
        lst_dst = []
        for edge in g.edges:
            lst_src.append(nodes_map[edge[0]])
            lst_dst.append(nodes_map[edge[1]])
        return torch.tensor([lst_src, lst_dst])


    @staticmethod
    def get_data_division(num_nodes):
        train_data = [False] * num_nodes
        validation_data = [False] * num_nodes
        test_data = [False] * num_nodes
        indices = set(range(0, num_nodes))
        train_idx = random.sample(indices, int(0.7 * num_nodes))
        indices = indices.difference(train_idx)
        validation_idx = random.sample(indices, int(0.15 * num_nodes))
        indices = indices.difference(validation_idx)
        test_idx = list(indices)
        for idx in train_idx:
            train_data[idx] = True
        for idx in validation_idx:
            validation_data[idx] = True
        for idx in test_idx:
            test_data[idx] = True

        return train_data, validation_data, test_data

    @staticmethod
    def get_graphs():
        # G = nx.complete_graph(20)
        # orig_edges = set(G.edges())
        # to_remove = random.sample(orig_edges, 30)
        # edges = orig_edges.difference(to_remove)
        # random_graph = nx.Graph(edges).to_directed()

        edges_g1 = {('v0', 'v1'), ('v0', 'v2'), ('v1', 'v2'), ('v3', 'v2'), ('v1', 'v3'), ('v0', 'v4'), ('v1', 'v5'),
                    ('v6', 'v7'), ('v8', 'v3'), ('v5', 'v9'), ('v10', 'v8')}
        g1 = nx.Graph(edges_g1).to_directed()

        # edges_g1 = [('v0', 'v1'), ('v0', 'v2'), ('v1', 'v0'), ('v1', 'v2'), ('v2', 'v0'), ('v2', 'v1')]
        # g1 = nx.DiGraph(edges_g1)
        # edges_g2 = [('v0', 'v1'), ('v1', 'v2'), ('v1', 'v0'), ('v2', 'v1')]
        # g2 = nx.DiGraph(edges_g2)

        # edges_g4 = []
        # g4 = nx.DiGraph(edges_g4)
        # g4.add_nodes_from(['v0', 'v1', 'v2'])
        #
        # edges_g5 = [('v0', 'v1'), ('v1', 'v0')]
        # g5 = nx.DiGraph(edges_g5)
        # g5.add_nodes_from(['v2'])
        #
        # edges_g6 = [('v1', 'v2'), ('v2', 'v1')]
        # g6 = nx.DiGraph(edges_g6)
        # g6.add_nodes_from(['v0'])
        #
        # edges_g7 = [('v0', 'v2'), ('v2', 'v0')]
        # g7 = nx.DiGraph(edges_g7)
        # g7.add_nodes_from(['v1'])

        # edges_g3 = [('v0', 'v1'), ('v1', 'v0'), ('v1', 'v2'), ('v2', 'v1'), ('v2', 'v1'), ('v3', 'v2'), ('v2', 'v3')]
        # g3 = nx.DiGraph(edges_g3)
        # g3.add_nodes_from(['v4', 'v5'])
        return [g1]


if __name__ == '__main__':
    data_set = MyOwnDataset(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset')
    data_set.process()
    a = 1
