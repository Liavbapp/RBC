import datetime
import os
import json

import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot, show
from Components.RBC_ML.RbcML import learn_models
import Components.RBC_REG.RBC as RBC_REG

DEVICE = torch.device('cuda:0')
DTYPE = torch.float


def test_spbc_on_graph(g, adj_matrix, test_num):
    print(f'started testing figure_{test_num}')
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    betweenness_tensor = get_betweenness_tensor(g, nodes_mapping)
    hyper_params = {'learning_rate': 1e-4,
                    'epochs': 4,
                    'momentum': 0.00,
                    'optimizer_type': 'sgd',
                    'pi_max_err': 0.0001}
    start_time = datetime.datetime.now()
    t_model, r_model = learn_models(adj_matrix, betweenness_tensor, hyper_params)
    rtime = datetime.datetime.now() - start_time
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC_REG.compute_rbc(g, r_model, t_model)
    print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
    save_info(adj_matrix, r_model, t_model, test_num, hyper_params, rtime)
    a = 1


def get_betweenness_tensor(g, nodes_mapping):
    tensor_raw = torch.tensor(list(nx.betweenness_centrality(g, endpoints=True).values()), dtype=DTYPE, device=DEVICE)
    tensor_norm = tensor_raw.clone()
    for node_val, node_idx in nodes_mapping.items():
        tensor_norm[node_val] = tensor_raw[node_idx]

    return tensor_norm


def test_spbc():
    edge_lst_easy = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edge_lst_easy_2 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)]
    edge_lst_0 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5)]
    edge_lst_0_1 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    edge_lst_1 = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 5)]
    edge_lst_2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
    edge_lst_3 = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4), (0, 4), (4, 5)]
    edge_lst_4 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4)]
    edge_lst_5 = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4), (5, 7),
                  (6, 7),
                  (4, 8), (7, 8), (8, 9), (4, 9)]
    # all_edge_lst = [edge_lst_0, edge_lst_0_1, edge_lst_1, edge_lst_2, edge_lst_3, edge_lst_4, edge_lst_5]
    all_edge_lst = [edge_lst_5]
    #
    # i = 0
    # for edge_lst in all_edge_lst:
    #     i += 1
    #     g = nx.Graph(edge_lst)
    #     plt.figure(i)
    #     nx.draw(g, with_labels=True)
    # plt.show()

    i = 0
    for edge_lst in all_edge_lst:
        i += 1
        g = nx.Graph(edge_lst)
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=DTYPE)
        test_spbc_on_graph(g, adj_matrix, test_num=i)


def draw_routing(routing_policy, s, t):
    routing_matrix_t = routing_policy[s, t].t()
    edges = [np.array(row.detach().to(device=torch.device('cpu'))) for row in list(torch.nonzero(routing_matrix_t))]
    edges_weights = [(i, j, {'weight': routing_matrix_t[i, j].item()}) for i, j in edges]
    g = nx.DiGraph(edges_weights)
    g.add_nodes_from(list(range(0, len(routing_matrix_t[0]))))
    # edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx(g, arrows=True)
    show()


def save_info(adj_matrix, routing_policy, traffic_matrix, test_num, hyper_params, rtime):
    adj_matrix_np = adj_matrix.detach().to(device=torch.device('cpu')).numpy()
    routing_policy_np = routing_policy.detach().to(device=torch.device('cpu')).numpy()
    traffic_matrix_np = traffic_matrix.detach().to(device=torch.device('cpu')).numpy()

    path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\{test_num}'
    if not os.path.isdir(path):
        os.mkdir(path)

    all_dirs = [int(name) for name in os.listdir(path) if os.path.isdir(path + '\\' + name)]
    if len(all_dirs) == 0:
        next_dir = 0
    else:
        next_dir = max(all_dirs) + 1

    dir_path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\{test_num}\\{str(next_dir)}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    np.save(f'{dir_path}\\adj_mat', adj_matrix_np)
    np.save(f'{dir_path}\\routing_policy', routing_policy_np)
    np.save(f'{dir_path}\\traffic_mat', traffic_matrix_np)
    with open(f'{dir_path}\\hyper_params.json', 'w') as fp:
        json.dump(hyper_params, fp)
        f = open(f'{dir_path}\\rtime.txt', "w+")
        f.write(f'Runtime: {rtime}')
        f.close()


def load_info(path):
    path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\1\9'
    adj_matrix = np.load(path + '\\adj_mat.npy')
    routing_policy = np.load(path + '\\routing_policy.npy')
    traffic_matrix = np.load(path + '\\traffic_mat.npy')
    # hyper_params_df = np.load(path)

    a = 1


if __name__ == '__main__':
    load_info('')
    test_spbc()
