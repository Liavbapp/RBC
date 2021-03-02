import json

import torch
import networkx as nx

from Tests.Tools import saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, StatisticsParams as Stas


def get_learning_target(centrality, g, nodes_mapping, device, dtype, adj_mat):
    if centrality == Centralities.SPBC:
        params = {'k': None, 'normalized': False, 'weight': None, 'endpoints': False, 'seed': None}
        tensor_raw = torch.tensor(list(nx.betweenness_centrality(g, k=params['k'], normalized=params['normalized'],
                                                                 weight=params['weight'],endpoints=params['endpoints'],
                                                                 seed=params['seed']).values()), dtype=dtype, device=device)
        learning_target = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            learning_target[node_val] = tensor_raw[node_idx]
    elif centrality == Centralities.Degree:
        params = {}
        learning_target = torch.sum(adj_mat, 1)
    elif centrality == Centralities.Eigenvector:
        params = {'max_iter': 100, 'tol': 1e-06, 'nstart': None, 'weight': None}
        tensor_raw = torch.tensor(list(nx.eigenvector_centrality(g, max_iter=params['max_iter'], tol=params['tol'],
                                                                 nstart=params['nstart'],
                                                                 weight=params['weight']).values()), dtype=dtype, device=device)
        learning_target = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            learning_target[node_val] = tensor_raw[node_idx]
    elif centrality == Centralities.Closeness:
        params = {'u': None, 'distance': None, 'wf_improved': True}
        tensor_raw = torch.tensor(list(nx.closeness_centrality(g, u=params['u'], distance=params['distance'],
                                                               wf_improved=params['wf_improved']).values()), dtype=dtype, device=device)
        learning_target = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            learning_target[node_val] = tensor_raw[node_idx]
    else:
        raise NotImplementedError

    return learning_target, params


class ParamsManager:
    def __init__(self, g, centrality):
        self.centrality = centrality
        device = TorchDevice.cpu
        dtype = TorchDtype.float
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=dtype)
        nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
        learning_target, centrality_params = get_learning_target(centrality, g, nodes_mapping, device, dtype, adj_matrix)
        centrality_params = json.dumps(centrality_params)

        self.hyper_params = {HyperParams.learning_rate: 1e-4,
                             HyperParams.epochs: 5,
                             HyperParams.momentum: 0,
                             HyperParams.optimizer: OptimizerTypes.RmsProp,
                             HyperParams.pi_max_err: 0.0001,
                             HyperParams.error_type: ErrorTypes.mse
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.centrality_params: centrality_params,
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
        self.params_statistics = None
        self.params_stuck_statics = None

    def prepare_params_statistics(self, t_model, r_model, final_error, rtime, rbc_pred, optimizer_params):
        params_statistic_dict = {RbcMatrices.adjacency_matrix: self.learning_params[LearningParams.adjacency_matrix],
                       RbcMatrices.routing_policy: r_model,
                       RbcMatrices.traffic_matrix: t_model,
                       Stas.centrality: self.centrality,
                       Stas.centrality_params: self.learning_params[LearningParams.centrality_params],
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
        self.params_statistics = params_statistic_dict

    def prepare_stuck_params_statistics(self, centrality, adj_matrix, learning_target, learning_params, err_msg, optimizer_params):
        stuck_params_statistic_dict = {Stas.centrality: centrality,
                                       RbcMatrices.adjacency_matrix: adj_matrix,
                                       Stas.target: learning_target,
                                       LearningParams.name: learning_params,
                                       Stas.comments: err_msg,
                                       Stas.optimizer_params: optimizer_params
                                       }
        self.params_stuck_statics = stuck_params_statistic_dict

    def save_params_statistics(self):
        if self.params_statistics is not None:
            saver.save_statistics(**self.params_statistics)
        else:
            saver.save_info_stuck(**self.params_stuck_statics)
