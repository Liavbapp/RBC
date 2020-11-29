import unittest
import networkx as nx
import Components.RBC_torch as RBC_T
import torch

edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
         ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
         ('v2', 't'), ('v5', 't'), ('v3', 't')}
g_1 = nx.DiGraph(edges)
nodes_map_g1 = {k: v for v, k in enumerate(list(g_1.nodes()))}
num_nodes_g1 = g_1.number_of_nodes()

edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
g_2 = nx.DiGraph(edges)
nodes_map_g2 = {k: v for v, k in enumerate(list(g_2.nodes()))}
num_nodes_g2 = g_2.number_of_nodes()


edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4'), ('v2', 'v1')}
g_3 = nx.DiGraph(edges)
nodes_map_g3 = {k: v for v, k in enumerate(list(g_3.nodes()))}
num_nodes_g3 = g_3.number_of_nodes()

edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4'), ('v2', 'v1'), ('v4', 'v2')}
g_4 = nx.DiGraph(edges)
nodes_map_g4 = {k: v for v, k in enumerate(list(g_4.nodes()))}
num_nodes_g4 = g_4.number_of_nodes()

# plt.figure(figsize=(15, 15))
# nx.draw_kamada_kawai(g_1, with_labels=True)
# plt.show()

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

    def test_rbc_g1_betweenness(self):
        pass
        # expected_res = {'v1': 400, 'v2': 700, 'v3': 600, 'v4': 700}
        # r_policy = RBC_T.get_betweenness_policy_tensor(g_2, nodes_map_g2)
        # t_val = torch.full(size=(num_nodes_g2, num_nodes_g2), fill_value=100, dtype=float)
        # rbc_tensor = RBC_T.rbc(g_2, r_policy, t_val)
        # for key, val in expected_res.items():
        #     self.assertTrue(rbc_tensor[nodes_map_g2[key]] == val)

    def test_rbc_g2_betweenness(self):
        expected_res = {'v1': 400, 'v2': 700, 'v3': 600, 'v4': 700}
        r_policy = RBC_T.get_betweenness_policy_tensor(g_2, nodes_map_g2)
        t_val = torch.full(size=(num_nodes_g2, num_nodes_g2), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.rbc(g_2, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g2[key]] == val)

    def test_rbc_g3_betweenness_loops(self):
        expected_res = {'v1': 500, 'v2': 700, 'v3': 600, 'v4': 700}
        r_policy = RBC_T.get_betweenness_policy_tensor(g_3, nodes_map_g3)
        t_val = torch.full(size=(num_nodes_g3, num_nodes_g3), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.rbc(g_3, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g3[key]] == val)

    def test_rbc_g_4_betweenness_loops(self):
        expected_res = {'v1': 700, 'v2': 1200, 'v3': 700, 'v4': 900}
        r_policy = RBC_T.get_betweenness_policy_tensor(g_4, nodes_map_g4)
        t_val = torch.full(size=(num_nodes_g4, num_nodes_g4), fill_value=100, dtype=float)
        rbc_tensor = RBC_T.rbc(g_4, r_policy, t_val)
        for key, val in expected_res.items():
            self.assertTrue(rbc_tensor[nodes_map_g4[key]] == val)


