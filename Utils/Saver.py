import json

import numpy as np
import torch
import networkx as nx
from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
import os
import pandas as pd

from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import StatisticsParams as ParamsStats, EmbeddingStatistics as EmbStats, EmbeddingOutputs, \
    EigenvectorMethod
from Utils.CommonStr import RbcMatrices
from Utils.CommonStr import LearningParams
from Utils.CommonStr import HyperParams


def plot_orig_graph(adj_mat):
    g = nx.from_numpy_matrix(adj_mat)
    nx.draw(g, with_labels=True)
    plt.show()


def draw_routing(routing_policy, s, t):
    routing_matrix_t = np.transpose(routing_policy[s, t])
    edges = [np.array(row) for row in list(torch.nonzero(torch.tensor(routing_matrix_t)))]
    edges_weights = [(i, j, {'weight': routing_matrix_t[i, j].item()}) for i, j in edges]
    g = nx.DiGraph(edges_weights)
    g.add_nodes_from(list(range(0, len(routing_matrix_t[0]))))
    # edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx(g, arrows=True)
    show()


def save_info(**kwargs):
    rbc_matrices_root_path = kwargs[RbcMatrices.root_path]
    centrality = kwargs[ParamsStats.centrality]
    adj_matrix = kwargs[RbcMatrices.adjacency_matrix]
    saving_path = get_saving_matrix_path(centrality, adj_matrix, rbc_matrices_root_path)
    save_matrices(kwargs[RbcMatrices.adjacency_matrix], kwargs[RbcMatrices.routing_policy],
                  kwargs[RbcMatrices.traffic_matrix], saving_path)
    kwargs.update({ParamsStats.path: saving_path})
    if not kwargs['stuck']:
        save_statistics(**kwargs)
    else:
        save_info_stuck(**kwargs)


def save_info_embeddings(**kwargs):
    test_r_policy = kwargs[EmbeddingOutputs.test_routing_policy]
    trained_model = kwargs[EmbeddingOutputs.trained_model]
    # test_graph = kwargs[EmbeddingOutputs.test_graph]
    root_embedding_outputs_path = kwargs[EmbeddingOutputs.root_path]
    train_path_params = kwargs[EmbeddingOutputs.train_path_params]
    test_path_params = kwargs[EmbeddingOutputs.test_path_params]

    saving_path = get_saving_embedding_matrices_path(kwargs[EmbStats.centrality], root_embedding_outputs_path)
    save_embedding_outputs(test_r_policy, trained_model, train_path_params, test_path_params, saving_path)
    kwargs.update({EmbStats.path: saving_path})
    if not kwargs['stuck']:
        save_statistics_embeddings(**kwargs)
    else:
        save_info_stuck_embeddings(**kwargs)


def save_statistics(**kwargs):
    num_nodes = len(kwargs[RbcMatrices.adjacency_matrix][0])
    num_edges = len(torch.nonzero(torch.triu(kwargs[RbcMatrices.adjacency_matrix]), as_tuple=True)[0])
    cols = ParamsStats.cols
    new_statistics = {ParamsStats.id: kwargs[ParamsStats.id],
                      ParamsStats.centrality: kwargs[ParamsStats.centrality],
                      ParamsStats.centrality_params: kwargs[ParamsStats.centrality_params],
                      ParamsStats.num_nodes: num_nodes,
                      ParamsStats.num_edges: num_edges,
                      ParamsStats.target: str(kwargs[ParamsStats.target]),
                      ParamsStats.prediction: str(kwargs[ParamsStats.prediction]),
                      ParamsStats.error: kwargs[ParamsStats.error],
                      ParamsStats.error_type: kwargs[ParamsStats.error_type],
                      ParamsStats.sigmoid: kwargs[ParamsStats.sigmoid],
                      ParamsStats.src_src_one: kwargs[ParamsStats.src_src_one],
                      ParamsStats.src_row_zeros: kwargs[ParamsStats.src_row_zeros],
                      ParamsStats.target_col_zeros: kwargs[ParamsStats.target_col_zeros],
                      ParamsStats.fixed_T: False if kwargs[ParamsStats.fixed_T] is None else True,
                      ParamsStats.fixed_R: False if kwargs[ParamsStats.fixed_R] is None else True,
                      ParamsStats.runtime: str(kwargs[ParamsStats.runtime]),
                      ParamsStats.learning_rate: kwargs[ParamsStats.learning_rate],
                      ParamsStats.epochs: kwargs[ParamsStats.epochs],
                      ParamsStats.optimizer: kwargs[ParamsStats.optimizer],
                      ParamsStats.optimizer_params: kwargs[ParamsStats.optimizer_params],
                      ParamsStats.pi_max_err: kwargs[ParamsStats.pi_max_err],
                      ParamsStats.path: kwargs[ParamsStats.path],
                      ParamsStats.comments: None,
                      ParamsStats.eigenvector_method: kwargs[ParamsStats.eigenvector_method],
                      ParamsStats.device: kwargs[ParamsStats.device],
                      ParamsStats.dtype: kwargs[ParamsStats.dtype],
                      ParamsStats.consider_traffic_paths: kwargs[ParamsStats.consider_traffic_paths]}
    df_new_statistics = pd.DataFrame(new_statistics, index=[0])

    csv_path = kwargs[ParamsStats.csv_save_path]
    try:
        df_statistics_old = pd.read_csv(csv_path)
    except Exception as ex:
        df_statistics_old = pd.DataFrame(columns=cols)

    df_combined_statistics = pd.concat([df_statistics_old, df_new_statistics])
    df_combined_statistics.to_csv(csv_path, index=False)


def save_info_stuck(**kwargs):
    adj_matrix = kwargs[RbcMatrices.adjacency_matrix]
    learning_params = kwargs[LearningParams.name]
    centrality = kwargs[ParamsStats.centrality]
    num_nodes = len(adj_matrix[0])
    num_edges = len(torch.nonzero(torch.triu(adj_matrix), as_tuple=True)[0])
    cols = ParamsStats.cols
    new_statistics = {ParamsStats.id: kwargs[ParamsStats.id],
                      ParamsStats.centrality: centrality,
                      ParamsStats.centrality_params: learning_params[LearningParams.centrality_params],
                      ParamsStats.num_nodes: num_nodes,
                      ParamsStats.num_edges: num_edges,
                      ParamsStats.target: str(kwargs[ParamsStats.target]),
                      ParamsStats.prediction: None,
                      ParamsStats.error: None,
                      ParamsStats.error_type: None,
                      ParamsStats.sigmoid: learning_params[LearningParams.sigmoid],
                      ParamsStats.src_src_one: learning_params[LearningParams.src_src_one],
                      ParamsStats.src_row_zeros: learning_params[LearningParams.src_row_zeros],
                      ParamsStats.target_col_zeros: learning_params[LearningParams.target_col_zeros],
                      ParamsStats.runtime: None,
                      ParamsStats.learning_rate: learning_params[LearningParams.hyper_parameters][
                          HyperParams.learning_rate],
                      ParamsStats.epochs: learning_params[LearningParams.hyper_parameters][HyperParams.epochs],
                      ParamsStats.optimizer: learning_params[LearningParams.hyper_parameters][HyperParams.optimizer],
                      ParamsStats.optimizer_params: kwargs[ParamsStats.optimizer_params],
                      ParamsStats.pi_max_err: learning_params[LearningParams.hyper_parameters][HyperParams.pi_max_err],
                      ParamsStats.path: get_saving_matrix_path(centrality, adj_matrix),
                      ParamsStats.comments: kwargs[ParamsStats.comments],
                      ParamsStats.consider_traffic_paths: learning_params[LearningParams.consider_traffic_paths]}
    df_new_statistics = pd.DataFrame(new_statistics, index=[0])

    csv_path = kwargs[ParamsStats.csv_save_path]
    try:
        df_statistics_old = pd.read_csv(csv_path)
    except Exception as ex:
        df_statistics_old = pd.DataFrame(columns=cols)

    df_combined_statistics = pd.concat([df_statistics_old, df_new_statistics])
    df_combined_statistics.to_csv(csv_path, index=False)


def save_statistics_embeddings(**kwargs):
    cols = EmbStats.cols
    new_embed_statistics = {
        EmbStats.id: kwargs[EmbStats.id],
        EmbStats.centrality: kwargs[EmbStats.centrality],
        EmbStats.centrality_params: kwargs[EmbStats.centrality_params],
        EmbStats.n_graphs: kwargs[EmbStats.n_graphs],
        EmbStats.n_seeds_graph: kwargs[EmbStats.n_seeds_graph],
        EmbStats.routing_type: kwargs[EmbStats.routing_type],
        EmbStats.n_routing_graph: kwargs[EmbStats.n_routing_graph],
        EmbStats.n_random_samples_graph: kwargs[EmbStats.n_random_samples_graph],
        EmbStats.graphs_desc: kwargs[EmbStats.graphs_desc],
        EmbStats.embd_dim: kwargs[EmbStats.embd_dim],
        EmbStats.embedding_alg: kwargs[EmbStats.embedding_alg],
        EmbStats.rbc_target: str(kwargs[EmbStats.rbc_target]),
        EmbStats.rbc_test: str(kwargs[EmbStats.rbc_test]),
        EmbStats.train_error: kwargs[EmbStats.train_error],
        EmbStats.error_type: kwargs[EmbStats.error_type],
        EmbStats.network_structure: kwargs[EmbStats.network_structure],
        EmbStats.train_runtime: str(kwargs[EmbStats.train_runtime]),
        EmbStats.learning_rate: kwargs[EmbStats.learning_rate],
        EmbStats.epochs: kwargs[EmbStats.epochs],
        EmbStats.weight_decay: kwargs[EmbStats.weight_decay],
        EmbStats.batch_size: kwargs[EmbStats.batch_size],
        EmbStats.optimizer: kwargs[EmbStats.optimizer],
        EmbStats.optimizer_params: str(kwargs[EmbStats.optimizer_params]),
        EmbStats.pi_max_err: kwargs[EmbStats.pi_max_err],
        EmbStats.path: kwargs[EmbStats.path],
        EmbStats.comments: None,
        EmbStats.eigenvector_method: kwargs[EmbStats.eigenvector_method],
        EmbStats.device: kwargs[EmbStats.device],
        EmbStats.dtype: kwargs[EmbStats.dtype],
        EmbStats.rbc_diff: kwargs[EmbStats.rbc_diff]
    }
    df_new_embedding_statistics = pd.DataFrame(new_embed_statistics, index=[0])
    csv_path = kwargs[EmbStats.csv_save_path]

    try:
        df_statistics_old_embed = pd.read_csv(csv_path)
    except Exception as ex:
        df_statistics_old_embed = pd.DataFrame(columns=cols)

    df_combined_statistics_embedding = pd.concat([df_statistics_old_embed, df_new_embedding_statistics])[cols]
    df_combined_statistics_embedding.to_csv(csv_path, index=False)


def save_info_stuck_embeddings(**kwargs):
    centrality = kwargs[EmbStats.centrality]
    learning_params = kwargs[LearningParams.name]
    cols = EmbStats.cols

    new_statistics = {EmbStats.id: kwargs[EmbStats.id],
                      EmbStats.csv_save_path: kwargs[EmbStats.csv_save_path],
                      EmbStats.centrality: centrality,
                      EmbStats.centrality_params: learning_params[EmbStats.centrality_params],
                      EmbStats.embd_dim: kwargs[EmbStats.embd_dim],
                      EmbStats.rbc_target: str(kwargs[ParamsStats.target]),
                      EmbStats.rbc_test: None,
                      EmbStats.train_error: None,
                      EmbStats.error_type: None,
                      EmbStats.network_structure: kwargs[EmbStats.network_structure],
                      EmbStats.train_runtime: None,
                      EmbStats.learning_rate: learning_params[LearningParams.hyper_parameters][
                          HyperParams.learning_rate],
                      EmbStats.epochs: learning_params[LearningParams.hyper_parameters][HyperParams.epochs],
                      EmbStats.optimizer: learning_params[LearningParams.hyper_parameters][HyperParams.optimizer],
                      EmbStats.optimizer_params: kwargs[ParamsStats.optimizer_params],
                      EmbStats.pi_max_err: learning_params[LearningParams.hyper_parameters][HyperParams.pi_max_err],
                      EmbStats.path: None,
                      EmbStats.comments: kwargs[EmbStats.comments]}

    df_new_statistics = pd.DataFrame(new_statistics, index=[0])
    csv_path = kwargs[EmbStats.csv_save_path]

    try:
        df_statistics_old = pd.read_csv(csv_path)
    except Exception as ex:
        df_statistics_old = pd.DataFrame(columns=cols)

    df_combined_statistics = pd.concat([df_statistics_old, df_new_statistics])
    df_combined_statistics.to_csv(csv_path, index=False)


def get_saving_matrix_path(centrality, adj_matrix, rbc_matrices_root_path):
    num_nodes = len(adj_matrix[0])
    num_edges = int(torch.sum(adj_matrix).item() / 2)
    path = f'{rbc_matrices_root_path}\\{centrality}\\{str(num_nodes)}_nodes\\{str(num_edges)}_edges'
    if not os.path.isdir(path):
        os.makedirs(path)

    all_dirs = [int(name) for name in os.listdir(path) if os.path.isdir(path + '\\' + name)]
    next_dir = 0 if len(all_dirs) == 0 else max(all_dirs) + 1

    final_path = path + f'\\{str(next_dir)}'
    os.mkdir(final_path)

    return final_path


def get_saving_embedding_matrices_path(centrality, root_path):
    path = f'{root_path}\\{centrality}'
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


def save_embedding_outputs(test_routing_policy, trained_model, train_path_params, test_path_params, path):
    test_routing_policy_np = test_routing_policy.detach().to(device=torch.device('cpu')).numpy()
    np.save(f'{path}\\test_routing_policy', test_routing_policy_np)
    with open(f'{path}\\train_path_params.json', 'w') as f:
        json.dump(str(train_path_params), f)
    with open(f'{path}\\test_path_params.json', 'w') as f:
        json.dump(str(test_path_params), f)
    # np.save(f'{path}\\test_graph', test_graph)
    torch.save(trained_model.state_dict(), f'{path}\\trained_model.pt')


def load_info(path):
    adj_matrix = np.load(path + '\\adj_mat.npy')
    routing_policy = np.load(path + '\\routing_policy.npy')
    traffic_matrix = np.load(path + '\\traffic_mat.npy')
    return adj_matrix, routing_policy, traffic_matrix


if __name__ == '__main__':
    path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\Load\5_nodes\4_edges\0'
    device = torch.device('cuda:0')
    dtype = torch.float
    adj_mat, routing_policy, traffic_mat = load_info(path)
    routing_policy, traffic_mat = torch.tensor(routing_policy, device=device, dtype=dtype), torch.tensor(traffic_mat, device=device, dtype=dtype)
    rbc_handler = RBC(eigenvector_method=EigenvectorMethod.torch_eig, pi_max_error=0.0000, device=device, dtype=dtype)
    print(rbc_handler.compute_rbc(nx.convert_matrix.from_numpy_matrix(adj_mat), routing_policy, traffic_mat))
