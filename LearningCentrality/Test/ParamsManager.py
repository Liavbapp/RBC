import datetime
import json

import torch
import networkx as nx

from Utils import Saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, StatisticsParams as Stas, StatisticsParams


def get_learning_target(centrality, g, nodes_mapping, device, dtype, adj_mat):
    if centrality == Centralities.SPBC:
        params = {'k': None, 'normalized': False, 'weight': None, 'endpoints': True, 'seed': None}
        tensor_raw = torch.tensor(list(nx.betweenness_centrality(g, k=params['k'], normalized=params['normalized'],
                                                                 weight=params['weight'], endpoints=params['endpoints'],
                                                                 seed=params['seed']).values()), dtype=dtype,
                                  device=device)
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
                                                                 weight=params['weight']).values()), dtype=dtype,
                                  device=device)
        learning_target = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            learning_target[node_val] = tensor_raw[node_idx]
    elif centrality == Centralities.Closeness:
        params = {'u': None, 'distance': None, 'wf_improved': True}
        tensor_raw = torch.tensor(list(nx.closeness_centrality(g, u=params['u'], distance=params['distance'],
                                                               wf_improved=params['wf_improved']).values()),
                                  dtype=dtype, device=device)
        learning_target = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            learning_target[node_val] = tensor_raw[node_idx]
    else:
        raise NotImplementedError

    return learning_target, params


class ParamsManager:
    def __init__(self, g, params_dict):
        self.centrality = params_dict[StatisticsParams.centrality]
        self.csv_save_path = params_dict[StatisticsParams.csv_save_path]
        self.rbc_matrices_path = params_dict[RbcMatrices.root_path]
        device = params_dict[StatisticsParams.device]
        dtype = params_dict[StatisticsParams.dtype]
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=dtype)
        nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
        learning_target, centrality_params = get_learning_target(self.centrality, g, nodes_mapping, device, dtype,
                                                                 adj_matrix)
        centrality_params = json.dumps(centrality_params)

        self.hyper_params = {HyperParams.learning_rate: params_dict[HyperParams.learning_rate],
                             HyperParams.epochs: params_dict[HyperParams.epochs],
                             HyperParams.momentum: params_dict[HyperParams.momentum],
                             HyperParams.optimizer: params_dict[HyperParams.optimizer],
                             HyperParams.pi_max_err: params_dict[HyperParams.pi_max_err],
                             HyperParams.error_type: params_dict[HyperParams.error_type]
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.centrality_params: centrality_params,
                                LearningParams.adjacency_matrix: adj_matrix,
                                LearningParams.target: learning_target,
                                LearningParams.src_src_one: params_dict[LearningParams.src_src_one],
                                LearningParams.src_row_zeros: params_dict[LearningParams.src_row_zeros],
                                LearningParams.target_col_zeros: params_dict[LearningParams.target_col_zeros],
                                LearningParams.sigmoid: params_dict[LearningParams.sigmoid],
                                LearningParams.fixed_R: params_dict[LearningParams.fixed_R],
                                LearningParams.fixed_T: params_dict[LearningParams.fixed_T],
                                LearningParams.consider_traffic_paths: params_dict[LearningParams.consider_traffic_paths],
                                LearningParams.eigenvector_method: params_dict[LearningParams.eigenvector_method],
                                LearningParams.device: params_dict[StatisticsParams.device],
                                LearningParams.dtype: params_dict[StatisticsParams.dtype]
                                }
        self.params_statistics = None
        self.params_stuck_statics = None
        self.t_model = None
        self.r_model = None
        self.final_error = None
        self.rtime = None
        self.rbc_pred = None
        self.optimizer_params = None

    def prepare_params_statistics(self):
        params_statistic_dict = {Stas.id: datetime.datetime.now(),
                                 RbcMatrices.adjacency_matrix: self.learning_params[LearningParams.adjacency_matrix],
                                 RbcMatrices.routing_policy: self.r_model,
                                 RbcMatrices.traffic_matrix: self.t_model,
                                 RbcMatrices.root_path: self.rbc_matrices_path,
                                 Stas.csv_save_path: self.csv_save_path,
                                 Stas.centrality: self.centrality,
                                 Stas.centrality_params: self.learning_params[LearningParams.centrality_params],
                                 Stas.target: self.learning_params[LearningParams.target],
                                 Stas.prediction: self.rbc_pred,
                                 Stas.error: self.final_error,
                                 Stas.error_type: self.learning_params[LearningParams.hyper_parameters][
                                     HyperParams.error_type],
                                 Stas.sigmoid: self.learning_params[LearningParams.sigmoid],
                                 Stas.src_src_one: self.learning_params[LearningParams.src_src_one],
                                 Stas.src_row_zeros: self.learning_params[LearningParams.src_row_zeros],
                                 Stas.target_col_zeros: self.learning_params[LearningParams.target_col_zeros],
                                 Stas.fixed_R: self.learning_params[LearningParams.fixed_R],
                                 Stas.fixed_T: self.learning_params[LearningParams.fixed_T],
                                 Stas.runtime: self.rtime,
                                 Stas.learning_rate: self.hyper_params[Stas.learning_rate],
                                 Stas.epochs: self.hyper_params[Stas.epochs],
                                 Stas.optimizer: self.hyper_params[Stas.optimizer],
                                 Stas.optimizer_params: self.optimizer_params,
                                 Stas.pi_max_err: self.hyper_params[Stas.pi_max_err],
                                 Stas.eigenvector_method: self.learning_params[LearningParams.eigenvector_method],
                                 Stas.device: self.learning_params[LearningParams.device],
                                 Stas.dtype: self.learning_params[LearningParams.dtype],
                                 Stas.consider_traffic_paths: self.learning_params[
                                     LearningParams.consider_traffic_paths]
                                 }
        self.params_statistics = params_statistic_dict

    def prepare_stuck_params_statistics(self, centrality, adj_matrix, learning_target, learning_params, err_msg):
        stuck_params_statistic_dict = {Stas.id: datetime.datetime.now(),
                                       Stas.csv_save_path: self.csv_save_path,
                                       RbcMatrices.root_path: self.rbc_matrices_path,
                                       Stas.centrality: centrality,
                                       RbcMatrices.adjacency_matrix: adj_matrix,
                                       Stas.target: learning_target,
                                       LearningParams.name: learning_params,
                                       Stas.comments: err_msg,
                                       OptimizerTypes: self.optimizer_params,
                                       }
        self.params_stuck_statics = stuck_params_statistic_dict

    def save_params_statistics(self):
        if self.params_statistics is not None:
            self.params_statistics.update({'stuck': False})
            Saver.save_info(**self.params_statistics)
        else:
            Saver.save_info(**self.params_stuck_statics, stuck=True)
            self.params_statistics.update({'stuck': True})

