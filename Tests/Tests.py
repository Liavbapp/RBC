import unittest
import networkx as nx
import Components.RBC as RBC
import matplotlib.pyplot as plt
import scipy

edges = {('s', 'v1'), ('s', 'v4'), ('v1', 'v5'),
         ('v4', 'v5'), ('v1', 'v2'), ('v2', 'v3'),
         ('v2', 't'), ('v5', 't'), ('v3', 't')}
g_1 = nx.DiGraph(edges)

edges = {('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4')}
g_2 = nx.DiGraph(edges)


# plt.figure(figsize=(15, 15))
# nx.draw_kamada_kawai(g_1, with_labels=True)
# plt.show()

class Tests(unittest.TestCase):

    def test_get_betweenness_policy_dict_1(self):
        actual_policy = RBC.get_betweenness_policy_dict(g_1)
        expected_policy_edge_s_t = {('v4', 'v5'): 1 / 3, ('v1', 'v5'): 1 / 3, ('s', 'v4'): 1 / 3, ('v2', 't'): 1 / 3,
                                    ('s', 'v1'): 2 / 3, ('v1', 'v2'): 1 / 3, ('v5', 't'): 2 / 3}
        self.assertTrue(actual_policy[('s', 't')] == expected_policy_edge_s_t)  # existing path
        self.assertTrue(actual_policy[('s', 's')] == {})  # path of len 0
        self.assertTrue(actual_policy[('t', 's')] == {})  # non existing path

    def test_get_betweenness_policy_dict_2(self):
        actual_policy = RBC.get_betweenness_policy_dict(g_2)
        expected_policy = {('v3', 'v3'): {}, ('v3', 'v4'): {('v3', 'v4'): 1.0}, ('v3', 'v2'): {},
                           ('v3', 'v1'): {}, ('v4', 'v3'): {}, ('v4', 'v4'): {}, ('v4', 'v2'): {},
                           ('v4', 'v1'): {}, ('v2', 'v3'): {('v2', 'v3'): 1.0}, ('v2', 'v4'): {('v2', 'v4'): 1.0},
                           ('v2', 'v2'): {}, ('v2', 'v1'): {}, ('v1', 'v3'): {('v2', 'v3'): 1.0, ('v1', 'v2'): 1.0},
                           ('v1', 'v4'): {('v2', 'v4'): 1.0, ('v1', 'v2'): 1.0}, ('v1', 'v2'): {('v1', 'v2'): 1.0},
                           ('v1', 'v1'): {}}
        self.assertTrue(actual_policy == expected_policy)

    def test_rbc_1(self):
        expected_res = {'v1': 400, 'v2': 700, 'v3': 600, 'v4': 700}
        policy = {('v3', 'v3'): {}, ('v3', 'v4'): {('v3', 'v4'): 1.0}, ('v3', 'v2'): {},
                  ('v3', 'v1'): {}, ('v4', 'v3'): {}, ('v4', 'v4'): {}, ('v4', 'v2'): {},
                  ('v4', 'v1'): {}, ('v2', 'v3'): {('v2', 'v3'): 1.0}, ('v2', 'v4'): {('v2', 'v4'): 1.0},
                  ('v2', 'v2'): {}, ('v2', 'v1'): {}, ('v1', 'v3'): {('v2', 'v3'): 1.0, ('v1', 'v2'): 1.0},
                  ('v1', 'v4'): {('v2', 'v4'): 1.0, ('v1', 'v2'): 1.0}, ('v1', 'v2'): {('v1', 'v2'): 1.0},
                  ('v1', 'v1'): {}}
        actual_res = RBC.rbc(g_2, lambda s, u, v, t: policy[(s, t)][(u, v)] if (u, v) in policy[(s, t)] else 0,
                             lambda s, t: 100)
        rbc_arr = actual_res[0]
        node_mapping = actual_res[1]
        actual_res_dict = {}
        for node, arr_idx in node_mapping.items():
            actual_res_dict.update({node: rbc_arr[arr_idx]})
        self.assertTrue(expected_res == actual_res_dict)

    def test_rbc_2(self):
        expected_v2_res = 1150 + (200/9)
        policy = {('v2', 'v2'): {}, ('v2', 'v3'): {('v2', 'v3'): 1.0}, ('v2', 'v1'): {}, ('v2', 't'): {('v2', 't'): 1.0},
                  ('v2', 'v5'): {}, ('v2', 's'): {}, ('v2', 'v4'): {}, ('v3', 'v2'): {}, ('v3', 'v3'): {},
                  ('v3', 'v1'): {}, ('v3', 't'): {('v3', 't'): 1.0}, ('v3', 'v5'): {}, ('v3', 's'): {},
                  ('v3', 'v4'): {}, ('v1', 'v2'): {('v1', 'v2'): 1.0},
                  ('v1', 'v3'): {('v2', 'v3'): 1.0, ('v1', 'v2'): 1.0}, ('v1', 'v1'): {},
                  ('v1', 't'): {('v1', 'v2'): 0.5, ('v2', 't'): 0.5, ('v5', 't'): 0.5, ('v1', 'v5'): 0.5},
                  ('v1', 'v5'): {('v1', 'v5'): 1.0}, ('v1', 's'): {}, ('v1', 'v4'): {}, ('t', 'v2'): {},
                  ('t', 'v3'): {}, ('t', 'v1'): {}, ('t', 't'): {}, ('t', 'v5'): {}, ('t', 's'): {},
                  ('t', 'v4'): {}, ('v5', 'v2'): {}, ('v5', 'v3'): {}, ('v5', 'v1'): {}, ('v5', 't'): {('v5', 't'): 1.0},
                  ('v5', 'v5'): {}, ('v5', 's'): {}, ('v5', 'v4'): {}, ('s', 'v2'): {('v1', 'v2'): 1.0, ('s', 'v1'): 1.0},
                  ('s', 'v3'): {('v2', 'v3'): 1.0, ('v1', 'v2'): 1.0, ('s', 'v1'): 1.0}, ('s', 'v1'): {('s', 'v1'): 1.0},
                  ('s', 't'): {('v1', 'v2'): float(1.0)/3, ('v2', 't'): float(1.0)/3, ('v5', 't'): 2.0/3, ('s', 'v4'): 1.0/3,
                               ('v4', 'v5'): 1.0/3, ('v1', 'v5'): 1.0/3, ('s', 'v1'): 2.0/3},
                  ('s', 'v5'): {('s', 'v4'): 0.5, ('v4', 'v5'): 0.5, ('v1', 'v5'): 0.5, ('s', 'v1'): 0.5},
                  ('s', 's'): {}, ('s', 'v4'): {('s', 'v4'): 1.0}, ('v4', 'v2'): {}, ('v4', 'v3'): {}, ('v4', 'v1'): {},
                  ('v4', 't'): {('v4', 'v5'): 1.0, ('v5', 't'): 1.0}, ('v4', 'v5'): {('v4', 'v5'): 1.0},
                  ('v4', 's'): {}, ('v4', 'v4'): {}}
        actual_res = RBC.rbc(g_1, lambda s, u, v, t: policy[(s, t)][(u, v)] if (u, v) in policy[(s, t)] else 0,
                             lambda s, t: 100)
        rbc_arr = actual_res[0]
        nodes_mapping = actual_res[1]
        self.assertTrue(rbc_arr[nodes_mapping['v2']] == expected_v2_res)
