import unittest
import networkx as nx
import Components.RBC_REG.RBC as RBC_T
import Components.RBC_REG.Policy as Policy
import torch

edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
         ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
         ('v2', 't'), ('v5', 't'), ('v3', 't')}
g1_dag = nx.DiGraph(edges)
nodes_map_g1_dag = {k: v for v, k in enumerate(list(g1_dag.nodes()))}
num_nodes_g1_dag = g1_dag.number_of_nodes()

edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
g2_dag = nx.DiGraph(edges)
nodes_map_g2_dag = {k: v for v, k in enumerate(list(g2_dag.nodes()))}
num_nodes_g2_dag = g2_dag.number_of_nodes()


edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4'), ('v2', 'v1')}
g3_directed = nx.DiGraph(edges)
nodes_map_g3_directed = {k: v for v, k in enumerate(list(g3_directed.nodes()))}
num_nodes_g3_directed = g3_directed.number_of_nodes()

edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4'), ('v2', 'v1'), ('v4', 'v2')}
g4_directed = nx.DiGraph(edges)
nodes_map_g4_directed = {k: v for v, k in enumerate(list(g4_directed.nodes()))}
num_nodes_g4_directed = g4_directed.number_of_nodes()

edges = {('v1', 'v2')}
g1 = nx.Graph(edges)
node_map_g1 = {k: v for v, k in enumerate(list(g1.nodes()))}
num_nodes_g1 = g1.number_of_nodes()

edges = {('v0', 'v1'), ('v1', 'v2'), ('v2', 'v3'), ('v3', 'v4'), ('v1', 'v5'), ('v1', 'v6'), ('v4', 'v5'), ('v5', 'v6')}
g2 = nx.Graph(edges)
node_map_g2 = {k: v for v, k in enumerate(list(g2.nodes()))}
num_nodes_g2 = g2.number_of_nodes()

edges = {('v0', 'v1'), ('v1', 'v2')}
g3 = nx.Graph(edges)
g3.add_nodes_from(['v3', 'v4', 'v5', 'v6'])
nodes_map_g3 = {k: v for v, k in enumerate(list(g3.nodes()))}
num_nodes_g3 = g3.number_of_nodes()

g4 = nx.Graph()
g4.add_nodes_from(['v1', 'v2', 'v3'])
nodes_map_g4 = {k: v for v, k in enumerate(list(g4.nodes()))}
num_nodes_g4 = g4.number_of_nodes()


betweenness_policy = Policy.BetweennessPolicy()
degree_policy = Policy.DegreePolicy()

class Tests(unittest.TestCase):
    #
    # def test_get_betweenness_policy_dict_1(self):
    #     actual_policy = RBC.get_betweenness_policy_dict(g_1)
    #     expected_policy_edge_s_t = {('v4', 'v5'): 1 / 3, ('v1', 'v5'): 1 / 3, ('s', 'v4'): 1 / 3, ('v2', 't'): 1 / 3,
    #                                 ('s', 'v1'): 2 / 3, ('v1', 'v2'): 1 / 3, ('v5', 't'): 2 / 3}
    #     self.assertTrue(actual_policy[('s', 't')] == expected_policy_edge_s_t)  # existing path
    #     self.assertTrue(actual_policy[('s', 's')] == {})  # path of len 0
    #     self.assertTrue(actual_policy[('t', 's')] == {})  # non existing path
    #
    # def test_get_betweenness_policy_dict_2(self):
    #     actual_policy = RBC.get_betweenness_policy_dict(g_2)
    #     expected_policy = {('v3', 'v3'): {}, ('v3', 'v4'): {('v3', 'v4'): 1.0}, ('v3', 'v2'): {},
    #                        ('v3', 'v1'): {}, ('v4', 'v3'): {}, ('v4', 'v4'): {}, ('v4', 'v2'): {},
    #                        ('v4', 'v1'): {}, ('v2', 'v3'): {('v2', 'v3'): 1.0}, ('v2', 'v4'): {('v2', 'v4'): 1.0},
    #                        ('v2', 'v2'): {}, ('v2', 'v1'): {}, ('v1', 'v3'): {('v2', 'v3'): 1.0, ('v1', 'v2'): 1.0},
    #                        ('v1', 'v4'): {('v2', 'v4'): 1.0, ('v1', 'v2'): 1.0}, ('v1', 'v2'): {('v1', 'v2'): 1.0},
    #                        ('v1', 'v1'): {}}
    #     self.assertTrue(actual_policy == expected_policy)

    def test_rbc_g1_dag_betweenness(self):
        pass
        # expected_res = {'v1': 400, 'v2': 700, 'v3': 600, 'v4': 700}
        # r_policy = RBC_T.get_betweenness_policy_tensor(g_2, nodes_map_g2)
        # t_val = torch.full(size=(num_nodes_g2, num_nodes_g2), fill_value=100, dtype=float)
        # rbc_tensor = RBC_T.rbc(g_2, r_policy, t_val)
        # for key, val in expected_res.items():
        #     self.assertTrue(rbc_tensor[nodes_map_g2[key]] == val)

    def test_rbc_g2_dag_betweenness(self):
        expected_res = {'v1': 400, 'v2': 700, 'v3': 600, 'v4': 700}
        r_policy = betweenness_policy.get_policy_tensor(g2_dag, nodes_map_g2_dag)
        t_val = torch.full(size=(num_nodes_g2_dag, num_nodes_g2_dag), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.compute_rbc(g2_dag, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g2_dag[key]] == val)

    def test_rbc_g3_directed_betweenness_loops(self):
        expected_res = {'v1': 500, 'v2': 700, 'v3': 600, 'v4': 700}
        r_policy = betweenness_policy.get_policy_tensor(g3_directed, nodes_map_g3_directed)
        t_val = torch.full(size=(num_nodes_g3_directed, num_nodes_g3_directed), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.compute_rbc(g3_directed, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g3_directed[key]] == val)

    def test_rbc_g_4_directed_betweenness_loops(self):
        expected_res = {'v1': 700, 'v2': 1200, 'v3': 700, 'v4': 900}
        r_policy = betweenness_policy.get_policy_tensor(g4_directed, nodes_map_g4_directed)
        t_val = torch.full(size=(num_nodes_g4_directed, num_nodes_g4_directed), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.compute_rbc(g4_directed, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g4_directed[key]] == val)

    def test_rbc_degree_g1(self):
        expected_res = {'v1': 1, 'v2': 1}
        deg_policy = degree_policy.get_policy_tensor(g1, node_map_g1)
        t_val = degree_policy.get_t_tensor(g1)
        rbc_tensor = RBC_T.compute_rbc(g1, deg_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[node_map_g1[key]] == val)

    def test_rbc_degree_g2(self):
        expected_res = {'v0': 1, 'v1': 4, 'v2': 2, 'v3': 2, 'v4': 2, 'v5': 3, 'v6': 2}
        deg_policy = degree_policy.get_policy_tensor(g2, node_map_g2)
        t_val = degree_policy.get_t_tensor(g2)
        rbc_tensor = RBC_T.compute_rbc(g2, deg_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[node_map_g2[key]] == val)

    def test_rbc_degree_g3(self):
        expected_res = {'v0': 1, 'v1': 2, 'v2': 1, 'v3': 0, 'v4': 0, 'v5': 0, 'v6': 0}
        deg_policy = degree_policy.get_policy_tensor(g3, nodes_map_g3)
        t_val = degree_policy.get_t_tensor(g3)
        rbc_tensor = RBC_T.compute_rbc(g3, deg_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g3[key]] == val)

    def rbc_degree_g4(self):
        expected_res = {'v1': 0, 'v2': 0, 'v3': 0}
        deg_policy = degree_policy.get_policy_tensor(g4, nodes_map_g4)
        t_val = degree_policy.get_t_tensor(g4)
        rbc_tensor = RBC_T.compute_rbc(g4, deg_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g4[key]] == val)