import datetime
import json

import torch
import networkx as nx

from Utils import Saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, EmbeddingStatistics as EmbStas, EmbeddingOutputs


def get_centrality_params(centrality, device, dtype):
    #TODO: get centrality params from csv
    return ""


class EmbeddingsParams:
    def __init__(self, params_dict):
        self.centrality = params_dict[EmbStas.centrality]
        self.embedding_dimensions = params_dict[EmbStas.embedding_dimensions]
        self.device = params_dict[EmbStas.device]
        self.dtype = params_dict[EmbStas.dtype]
        self.network_structure = params_dict[EmbStas.network_structure]
        self.optimizer_params = params_dict[EmbStas.optimizer_params]
        centrality_params = get_centrality_params(self.centrality, self.device, self.dtype)

        self.hyper_params = {HyperParams.learning_rate: params_dict[HyperParams.learning_rate],
                             HyperParams.epochs: params_dict[HyperParams.epochs],
                             HyperParams.momentum: params_dict[HyperParams.momentum],
                             HyperParams.optimizer: params_dict[HyperParams.optimizer],
                             HyperParams.pi_max_err: params_dict[HyperParams.pi_max_err],
                             HyperParams.error_type: params_dict[HyperParams.error_type],
                             HyperParams.batch_size: params_dict[HyperParams.batch_size]
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.centrality_params: centrality_params,
                                LearningParams.eigenvector_method: params_dict[EmbStas.eigenvector_method],
                                LearningParams.device: self.device,
                                LearningParams.dtype: self.dtype
                                }
        self.trained_model = None
        self.train_runtime = None
        self.train_error = None
        self.expected_rbc = None
        self.actual_rbc = None
        self.emb_params_statistics = None
        self.emb_params_stuck_statics = None
        self.test_routing_policy = None
        self.test_graph = None

    def prepare_params_statistics(self):
        params_statistic_dict = {EmbStas.id: datetime.datetime.now(),
                                 EmbStas.centrality: self.centrality,
                                 EmbStas.centrality_params: self.learning_params[LearningParams.centrality_params],
                                 EmbStas.embedding_dimensions: self.embedding_dimensions,
                                 EmbStas.rbc_target: self.expected_rbc,
                                 EmbStas.rbc_test: self.actual_rbc,
                                 EmbStas.train_error: self.train_error,
                                 EmbStas.error_type: self.learning_params[LearningParams.hyper_parameters][HyperParams.error_type],
                                 EmbStas.train_runtime: self.train_runtime,
                                 EmbStas.network_structure: self.network_structure,
                                 EmbStas.learning_rate: self.hyper_params[EmbStas.learning_rate],
                                 EmbStas.epochs: self.hyper_params[EmbStas.epochs],
                                 EmbStas.batch_size: self.hyper_params[EmbStas.batch_size],
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
        stuck_params_statistic_dict = {EmbStas.id: datetime.datetime.now(),
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
            Saver.save_info_embeddings(**self.emb_params_stuck_statics, stuck=True)
            self.emb_params_statistics.update({'stuck': True})

