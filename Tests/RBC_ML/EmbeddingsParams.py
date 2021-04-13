import datetime
import json

import torch
import networkx as nx

from Utils import Saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, EmbeddingStatistics as EmbStas, EmbeddingOutputs


def get_centrality_params(centrality, device, dtype):
    # TODO: get centrality params from csv
    return ""


class EmbeddingsParams:
    def __init__(self, params_dict):
        self.centrality = params_dict[EmbStas.centrality]
        self.csv_path = params_dict[EmbStas.csv_save_path]
        self.embedding_output_root_path = params_dict[EmbeddingOutputs.root_path]
        self.embedding_dimensions = params_dict[EmbStas.embd_dim]
        self.device = params_dict[EmbStas.device]
        self.dtype = params_dict[EmbStas.dtype]
        self.seed_range = params_dict['seed_range']
        self.graph_paths = params_dict[EmbStas.graphs_desc].paths
        self.graphs_desc = params_dict[EmbStas.graphs_desc].graphs_desc
        self.embedding_alg_name = params_dict[EmbStas.embedding_alg]
        # self.graph_paths = params_dict['graph_paths']
        centrality_params = get_centrality_params(self.centrality, self.device, self.dtype)

        self.hyper_params = {HyperParams.learning_rate: params_dict[HyperParams.learning_rate],
                             HyperParams.epochs: params_dict[HyperParams.epochs],
                             HyperParams.momentum: params_dict[HyperParams.momentum],
                             HyperParams.optimizer: params_dict[HyperParams.optimizer],
                             HyperParams.pi_max_err: params_dict[HyperParams.pi_max_err],
                             HyperParams.error_type: params_dict[HyperParams.error_type],
                             HyperParams.batch_size: params_dict[HyperParams.batch_size],
                             HyperParams.weight_decay: params_dict[HyperParams.weight_decay]
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.centrality_params: centrality_params,
                                LearningParams.eigenvector_method: params_dict[EmbStas.eigenvector_method],
                                LearningParams.device: self.device,
                                LearningParams.dtype: self.dtype
                                }

        self.train_path_params = None
        self.test_path_params = None
        self.optimizer_params = None
        self.trained_model = None
        self.train_runtime = None
        self.train_error = None
        self.expected_rbc = None
        self.actual_rbc = None
        self.emb_params_statistics = None
        self.emb_params_stuck_statics = None
        self.test_routing_policy = None
        self.test_graph = None
        self.network_structure = None
        self.rbc_diff = None

    def prepare_params_statistics(self):
        params_statistic_dict = {EmbeddingOutputs.root_path: self.embedding_output_root_path,
                                 EmbeddingOutputs.train_path_params: self.train_path_params,
                                 EmbeddingOutputs.test_path_params: self.test_path_params,
                                 EmbStas.csv_save_path: self.csv_path,
                                 EmbStas.id: datetime.datetime.now(),
                                 EmbStas.centrality: self.centrality,
                                 EmbStas.centrality_params: self.learning_params[LearningParams.centrality_params],
                                 EmbStas.graphs_desc: self.graphs_desc,
                                 EmbStas.embedding_alg: self.embedding_alg_name,
                                 EmbStas.embd_dim: self.embedding_dimensions,
                                 EmbStas.rbc_target: self.expected_rbc,
                                 EmbStas.rbc_test: self.actual_rbc,
                                 EmbStas.train_error: self.train_error,
                                 EmbStas.error_type: self.learning_params[LearningParams.hyper_parameters][
                                     HyperParams.error_type],
                                 EmbStas.rbc_diff: self.rbc_diff,
                                 EmbStas.train_runtime: self.train_runtime,
                                 EmbStas.network_structure: self.network_structure,
                                 EmbStas.learning_rate: self.hyper_params[EmbStas.learning_rate],
                                 EmbStas.epochs: self.hyper_params[EmbStas.epochs],
                                 EmbStas.batch_size: self.hyper_params[EmbStas.batch_size],
                                 EmbStas.weight_decay: self.hyper_params[EmbStas.weight_decay],
                                 EmbStas.optimizer: self.hyper_params[EmbStas.optimizer],
                                 EmbStas.optimizer_params: self.optimizer_params,
                                 EmbStas.eigenvector_method: self.learning_params[LearningParams.eigenvector_method],
                                 EmbStas.pi_max_err: self.hyper_params[EmbStas.pi_max_err],
                                 EmbStas.device: self.learning_params[LearningParams.device],
                                 EmbStas.dtype: self.learning_params[LearningParams.dtype],
                                 EmbeddingOutputs.trained_model: self.trained_model,
                                 EmbeddingOutputs.test_routing_policy: self.test_routing_policy,
                                 EmbeddingOutputs.test_graph: self.test_graph

                                 }
        self.emb_params_statistics = params_statistic_dict

    def prepare_stuck_params_statistics(self, centrality, learning_target, learning_params, err_msg,
                                        optimizer_params):
        stuck_params_statistic_dict = {EmbeddingOutputs.train_path_params: self.train_path_params,
                                       EmbeddingOutputs.test_path_params: self.test_path_params,
                                       EmbStas.id: datetime.datetime.now(),
                                       EmbStas.csv_save_path: self.csv_path,
                                       EmbStas.centrality: centrality,
                                       EmbStas.rbc_target: learning_target,
                                       LearningParams.name: learning_params,
                                       EmbStas.comments: err_msg,
                                       OptimizerTypes: optimizer_params
                                       }
        self.emb_params_stuck_statics = stuck_params_statistic_dict

    def save_params_statistics(self):
        if self.emb_params_statistics is not None:
            self.emb_params_statistics.update({'stuck': False})
            Saver.save_info_embeddings(**self.emb_params_statistics)
        else:
            self.emb_params_statistics.update({'stuck': True})
            Saver.save_info_embeddings(**self.emb_params_stuck_statics, stuck=True)
