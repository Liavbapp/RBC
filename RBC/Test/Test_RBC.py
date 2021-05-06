import unittest
import networkx as nx
from RBC.RBC import RBC
import RBC.Policy as Policy
import torch
from Utils.CommonStr import EigenvectorMethod

edges = [(0, 1)]
g1 = nx.Graph(edges)

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (1, 6), (4, 5), (5, 6)]
g2 = nx.Graph(edges)

edges = [(0, 1), (1, 2)]
g3 = nx.Graph(edges)

g4 = nx.Graph()
g4.add_nodes_from([0, 1, 2])

edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
g6 = nx.Graph()
g6.add_edges_from(edges)

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
g7 = nx.Graph()
g7.add_edges_from(edges)

betweenness_policy = Policy.BetweennessPolicy()
degree_policy = Policy.DegreePolicy()


class Tests(unittest.TestCase):

    def init_object(self, eigenvector_method, max_pi_err=0.0001):
        self.rbc_handler = RBC(eigenvector_method, max_pi_err, torch.device('cpu'), torch.float)
        self.spbc_policy = Policy.BetweennessPolicy()
        self.degree_policy = Policy.DegreePolicy()

    def rbc_spbc(self, g, t_val=None):
        num_nodes = len(g.nodes())
        if t_val is None:
            t_val = torch.full(size=(num_nodes, num_nodes), fill_value=0.5, dtype=torch.float)
            t_val[torch.eye(num_nodes).byte()] = 0.0
        nodes_map_g = {k: v for v, k in enumerate(list(g.nodes()))}
        r_policy = betweenness_policy.get_policy_tensor(g, nodes_map_g)
        actual_rbc = self.rbc_handler.compute_rbc(g, r_policy, t_val)
        return actual_rbc, nodes_map_g

    def test_g1_pi_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        expected_rbc = nx.betweenness_centrality(g1, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g1)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g1_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        expected_rbc = nx.betweenness_centrality(g1, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g1)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g1_torch_eq_pi(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        pi_rbc = self.rbc_spbc(g1)[0]
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        eig_rbc = self.rbc_spbc(g1)[0]
        self.assertTrue(all(pi_rbc == eig_rbc))

    def test_g2_pi_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        expected_rbc = nx.betweenness_centrality(g2, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g2)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g2_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        expected_rbc = nx.betweenness_centrality(g2, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g2)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g2_torch_eq_pi(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        pi_rbc = self.rbc_spbc(g2)[0]
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        eig_rbc = self.rbc_spbc(g2)[0]
        self.assertTrue(all(pi_rbc == eig_rbc))

    def test_g3_pi_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        expected_rbc = nx.betweenness_centrality(g3, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g3)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g3_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        expected_rbc = nx.betweenness_centrality(g3, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g3)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g3_torch_eq_pi(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        pi_rbc = self.rbc_spbc(g3)[0]
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        eig_rbc = self.rbc_spbc(g3)[0]
        self.assertTrue(all(pi_rbc == eig_rbc))

    def test_g4_pi_spbc_1(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        t_val = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float)
        expected_rbc = nx.betweenness_centrality(g4, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g4, t_val)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g4_pi_spbc_2(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        t_val = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float)
        expected_rbc = nx.betweenness_centrality(g4, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g4, t_val)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g4_torch_spbc_1(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig, max_pi_err=0.00001)
        t_val = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float)
        expected_rbc = nx.betweenness_centrality(g4, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g4, t_val)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g4_torch_spbc_2(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig, max_pi_err=0.00001)
        t_val = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float)
        expected_rbc = nx.betweenness_centrality(g4, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g4, t_val)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g6_pi_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        expected_rbc = nx.betweenness_centrality(g6, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g6)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g6_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        expected_rbc = nx.betweenness_centrality(g6, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g6)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g6_eq_pi_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        pi_rbc = self.rbc_spbc(g6)[0]
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        eig_rbc = self.rbc_spbc(g6)[0]
        self.assertTrue(all(pi_rbc == eig_rbc))



    def test_g7_pi_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.00001)
        expected_rbc = nx.betweenness_centrality(g7, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g7)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g7_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        expected_rbc = nx.betweenness_centrality(g7, normalized=False, endpoints=True)
        actual_rbc, nodes_map_g = self.rbc_spbc(g7)
        for key, val in expected_rbc.items():
            self.assertTrue(actual_rbc[nodes_map_g[key]] == val)

    def test_g7_eq_pi_torch_spbc(self):
        self.init_object(eigenvector_method=EigenvectorMethod.power_iteration, max_pi_err=0.0001)
        pi_rbc = self.rbc_spbc(g7)[0]
        self.init_object(eigenvector_method=EigenvectorMethod.torch_eig)
        eig_rbc = self.rbc_spbc(g7)[0]
        self.assertTrue(all(pi_rbc == eig_rbc))
