import unittest
import networkx as nx
import torch

from Components.RBC_REG.Policy import LoadCentralityPolicy, BetweennessPolicy
from Components.RBC_REG.RBC import RBC
from Utils import Saver
from Utils.CommonStr import EigenvectorMethod, Centralities
from Utils.GraphGenerator import GraphGenerator


class Tests(unittest.TestCase):

    def test_betweeness_policy(self):
        graphs = GraphGenerator('SPBC').custom_graph()
        nodes_map_gs = [{k: v for v, k in enumerate(list(graph.nodes()))} for graph in graphs]
        betweenness_policy = BetweennessPolicy()
        Ts = [self.create_default_t_matrix(g.number_of_nodes()) for g in graphs]
        Rs = [betweenness_policy.get_policy_tensor(graph, nodes_map) for graph, nodes_map in zip(graphs, nodes_map_gs)]
        rbc_hanlder = RBC(EigenvectorMethod.power_iteration, 0.00001, torch.device('cpu'), torch.float)
        RBCs = [rbc_hanlder.compute_rbc(graph, r, t) for graph, r, t in zip(graphs, Rs, Ts)]
        i=0
        for graph, r, t in zip(graphs, Rs, Ts):
            print(f'{i} out of {len(graphs)}')
            i += 1
            adj_matx = torch.tensor(nx.adj_matrix(graph).todense(), dtype=torch.float)
            root_path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Experiments\\Experiments_1\\Data\\11_nodes_fixed_rbc\\Raw_Data\\SPBC'
            save_path = Saver.get_saving_matrix_path(Centralities.SPBC, adj_matx, root_path)
            Saver.save_matrices(adj_matx, r, t, save_path)

        print(RBCs)


    #
    # def test_load_centrality_policy(self):
    #     # graph = GraphGenerator.custom_graph()
    #     graphs = GraphGenerator('Load').custom_graph()
    #     nodes_map_gs = [{k: v for v, k in enumerate(list(graph.nodes()))} for graph in graphs]
    #     load_policies = [LoadCentralityPolicy(graph, nodes_map) for graph, nodes_map in zip(graphs, nodes_map_gs)]
    #     Rs = [load_policy.get_policy_tensor() for load_policy in load_policies]
    #     Ts = [torch.full(size=(graph.number_of_nodes(),) * 2, fill_value=0.5, dtype=torch.float) for graph in graphs]
    #     for t, graph in zip(Ts, graphs):
    #         t[torch.eye(graph.number_of_nodes()).byte()] = 0
    #     rbc_hanlder = RBC(EigenvectorMethod.torch_eig, 0.00001, torch.device('cpu'), torch.float)
    #     RBCs = [rbc_hanlder.compute_rbc(graph, r, t) for graph, r, t in zip(graphs, Rs, Ts)]
    #     i = 0
    #     for graph, r, t in zip(graphs, Rs, Ts):
    #         print(f'{i} out of {len(graphs)}')
    #         i += 1
    #         adj_matx = torch.tensor(nx.adj_matrix(graph).todense(), dtype=torch.float)
    #         root_path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results'
    #         save_path = Saver.get_saving_matrix_path(Centralities.Load, adj_matx, root_path)
    #         Saver.save_matrices(adj_matx, r, t, save_path)
    #
    #     print(RBCs)

        # expected_rbc = nx.betweenness_centrality(graph, normalized=False, endpoints=True)
        # actual_rbc, nodes_map_g = self.rbc_spbc(g1)
        # for key, val in expected_rbc.items():
        #     self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def create_default_t_matrix(self, num_nodes):
        t_mat = torch.full(size=(num_nodes, num_nodes), fill_value=0.5, dtype=torch.float)
        t_mat[torch.eye(num_nodes).byte()] = 0.0
        return t_mat
