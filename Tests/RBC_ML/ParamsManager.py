import torch
import networkx as nx

from Tests.Tools import saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, StatisticsParams as Stas


def get_learning_target(centrality, g, nodes_mapping, device, dtype, adj_mat):
    if centrality == Centralities.SPBC:
        tensor_raw = torch.tensor(list(nx.betweenness_centrality(g).values()), dtype=dtype, device=device)
        tensor_norm = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            tensor_norm[node_val] = tensor_raw[node_idx]
        return tensor_norm
    elif centrality == Centralities.Degree:
        return torch.sum(adj_mat, 1)
    elif centrality == Centralities.Eigenvector:
        tensor_raw = torch.tensor(list(nx.eigenvector_centrality(g).values()), dtype=dtype, device=device)
        tensor_norm = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            tensor_norm[node_val] = tensor_raw[node_idx]
        return tensor_norm
    elif centrality == Centralities.Closeness:
        tensor_raw = torch.tensor(list(nx.betweenness_centrality(g).values()), dtype=dtype, device=device)
        tensor_norm = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            tensor_norm[node_val] = tensor_raw[node_idx]
        return tensor_norm


class ParamsManager:
    def __init__(self, g, centrality):
        self.centrality = centrality
        device = TorchDevice.cpu
        dtype = TorchDtype.float
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=dtype)
        nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
        learning_target = get_learning_target(centrality, g, nodes_mapping, device, dtype, adj_matrix)

        self.hyper_params = {HyperParams.learning_rate: 1e-3,
                             HyperParams.epochs: 5,
                             HyperParams.momentum: 0,
                             HyperParams.optimizer: OptimizerTypes.RmsProp,
                             HyperParams.pi_max_err: 0.0001,
                             HyperParams.error_type: ErrorTypes.mse
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.adjacency_matrix: adj_matrix,
                                LearningParams.target: learning_target,
                                LearningParams.src_src_one: True,
                                LearningParams.src_row_zeros: False,
                                LearningParams.target_col_zeros: False,
                                LearningParams.sigmoid: True,
                                LearningParams.consider_traffic_paths: True,
                                LearningParams.eigenvector_method: EigenvectorMethod.power_iteration,
                                LearningParams.device: device,
                                LearningParams.dtype: dtype
                                }

    def save_params_statistics(self, t_model, r_model, final_error, rtime, rbc_pred, optimizer_params):
        kwargs_dict = {RbcMatrices.adjacency_matrix: self.learning_params[LearningParams.adjacency_matrix],
                       RbcMatrices.routing_policy: r_model,
                       RbcMatrices.traffic_matrix: t_model,
                       Stas.centrality: self.centrality,
                       Stas.target: self.learning_params[LearningParams.target],
                       Stas.prediction: rbc_pred,
                       Stas.error: final_error,
                       Stas.error_type: self.learning_params[LearningParams.hyper_parameters][HyperParams.error_type],
                       Stas.sigmoid: self.learning_params[LearningParams.sigmoid],
                       Stas.src_src_one: self.learning_params[LearningParams.src_src_one],
                       Stas.src_row_zeros: self.learning_params[LearningParams.src_row_zeros],
                       Stas.target_col_zeros: self.learning_params[LearningParams.target_col_zeros],
                       Stas.runtime: rtime,
                       Stas.learning_rate: self.hyper_params[Stas.learning_rate],
                       Stas.epochs: self.hyper_params[Stas.epochs],
                       Stas.optimizer: self.hyper_params[Stas.optimizer],
                       Stas.optimizer_params: optimizer_params,
                       Stas.pi_max_err: self.hyper_params[Stas.pi_max_err],
                       Stas.eigenvector_method: self.learning_params[LearningParams.eigenvector_method],
                       Stas.device: self.learning_params[LearningParams.device],
                       Stas.dtype: self.learning_params[LearningParams.dtype],
                       Stas.consider_traffic_paths: self.learning_params[LearningParams.consider_traffic_paths]
                       }

        saver.save_info(**kwargs_dict)
