import numpy as np
import torch
import networkx as nx
from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
import os
import pandas as pd
from Utils.CommonStr import StatisticsParams as Stas
from Utils.CommonStr import RbcMatrices


def plot_orig_graph(edge_lst, num_figure):
    plt.figure(num_figure)
    g = nx.Graph(edge_lst)
    nx.draw(g, with_labels=True)
    plt.show()


def draw_routing(routing_policy, s, t):
    routing_matrix_t = routing_policy[s, t].t()
    edges = [np.array(row.detach().to(device=torch.device('cpu'))) for row in list(torch.nonzero(routing_matrix_t))]
    edges_weights = [(i, j, {'weight': routing_matrix_t[i, j].item()}) for i, j in edges]
    g = nx.DiGraph(edges_weights)
    g.add_nodes_from(list(range(0, len(routing_matrix_t[0]))))
    # edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx(g, arrows=True)
    show()


def save_info(**kwargs):
    saving_path = get_saving_matrix_path(kwargs[Stas.centrality], kwargs[RbcMatrices.adjacency_matrix])
    save_matrices(kwargs[RbcMatrices.adjacency_matrix], kwargs[RbcMatrices.routing_policy],
                  kwargs[RbcMatrices.traffic_matrix], saving_path)
    kwargs.update({Stas.path: saving_path})
    save_statistics(**kwargs)


def save_statistics(**kwargs):
    num_nodes = len(kwargs[RbcMatrices.adjacency_matrix][0])
    num_edges = len(torch.nonzero(torch.triu(kwargs[RbcMatrices.adjacency_matrix]), as_tuple=True))
    cols = Stas.cols
    new_statistics = {Stas.centrality: kwargs[Stas.centrality],
                      Stas.num_nodes: num_nodes,
                      Stas.num_edges: num_edges,
                      Stas.target: str(kwargs[Stas.target]),
                      Stas.prediction: str(kwargs[Stas.prediction]),
                      Stas.error: kwargs[Stas.error],
                      Stas.error_type: kwargs[Stas.error_type],
                      Stas.sigmoid: kwargs[Stas.sigmoid],
                      Stas.src_src_one: kwargs[Stas.src_src_one],
                      Stas.src_row_zeros: kwargs[Stas.src_row_zeros],
                      Stas.target_col_zeros: kwargs[Stas.target_col_zeros],
                      Stas.runtime: str(kwargs[Stas.runtime]),
                      Stas.learning_rate: kwargs[Stas.learning_rate],
                      Stas.epochs: kwargs[Stas.epochs],
                      Stas.momentum: kwargs[Stas.momentum],
                      Stas.optimizer: kwargs[Stas.optimizer],
                      Stas.pi_max_err: kwargs[Stas.pi_max_err],
                      Stas.path: kwargs[Stas.path]}
    df_new_statistics = pd.DataFrame(new_statistics, index=[0])

    try:
        df_statistics_old = pd.read_csv(
            f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\Combined_Results\\statistics.csv')
    except Exception as ex:
        df_statistics_old = pd.DataFrame(columns=cols)

    df_combined_statistics = pd.concat([df_statistics_old, df_new_statistics])
    df_combined_statistics.to_csv(
        f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\Combined_Results\\statistics.csv', index=False)


def get_saving_matrix_path(centrality, adj_matrix):
    num_nodes = len(adj_matrix[0])
    num_edges = len(torch.nonzero(torch.triu(adj_matrix), as_tuple=True))
    path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\{centrality}\\' \
           f'{str(num_nodes)}_nodes\\{str(num_edges)}_edges'
    if not os.path.isdir(path):
        os.makedirs(path)

    all_dirs = [int(name) for name in os.listdir(path) if os.path.isdir(path + '\\' + name)]
    next_dir = 0 if len(all_dirs) == 0 else max(all_dirs) + 1

    final_path = path + f'\\{str(next_dir)}'
    os.mkdir(final_path)

    return final_path


def save_matrices(adj_matrix, routing_policy, traffic_matrix, path):
    adj_matrix_np = adj_matrix.detach().to(device=torch.device('cpu')).numpy()
    routing_policy_np = routing_policy.detach().to(device=torch.device('cpu')).numpy()
    traffic_matrix_np = traffic_matrix.detach().to(device=torch.device('cpu')).numpy()

    np.save(f'{path}\\adj_mat', adj_matrix_np)
    np.save(f'{path}\\routing_policy', routing_policy_np)
    np.save(f'{path}\\traffic_mat', traffic_matrix_np)


def load_info(path):
    path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\1\0'
    adj_matrix = np.load(path + '\\adj_mat.npy')
    routing_policy = np.load(path + '\\routing_policy.npy')
    traffic_matrix = np.load(path + '\\traffic_mat.npy')
    # hyper_params_df = np.load(path)

    a = 1