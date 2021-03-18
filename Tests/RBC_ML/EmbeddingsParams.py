import datetime
import json

import torch
import networkx as nx

from Utils import Saver
from Utils.CommonStr import TorchDevice, TorchDtype, HyperParams, LearningParams, EigenvectorMethod, OptimizerTypes, \
    ErrorTypes, Centralities, RbcMatrices, EmbeddingStatistics as EmbStas


def get_centrality_params(g, centrality, device, dtype):
    #TODO: get centrality params from csv
    return None


class EmbeddingsParams:
    def __init__(self,g, centrality):
        self.centrality = centrality
        device = TorchDevice.gpu
        dtype = TorchDtype.float
        centrality_params = get_centrality_params(g, centrality, device, dtype)

        self.hyper_params = {HyperParams.learning_rate: 1e-4,
                             HyperParams.epochs: 2000,
                             HyperParams.momentum: 0,
                             HyperParams.optimizer: OptimizerTypes.RmsProp,
                             HyperParams.pi_max_err: 0.00001,
                             HyperParams.error_type: ErrorTypes.mse
                             }
        self.learning_params = {LearningParams.hyper_parameters: self.hyper_params,
                                LearningParams.centrality_params: centrality_params,
                                LearningParams.eigenvector_method: EigenvectorMethod.power_iteration,
                                LearningParams.device: device,
                                LearningParams.dtype: dtype
                                }
        self.emb_params_statistics = None
        self.emb_params_stuck_statics = None

    def prepare_params_statistics(self, t_model, r_model, final_error, rtime, rbc_pred, optimizer_params):
        params_statistic_dict = {EmbStas.id: datetime.datetime.now(),
                                 EmbStas.centrality: self.centrality,
                                 EmbStas.centrality_params: self.learning_params[LearningParams.centrality_params],
                                 EmbStas.target: self.learning_params[LearningParams.target],
                                 EmbStas.prediction: rbc_pred,
                                 EmbStas.error: final_error,
                                 EmbStas.error_type: self.learning_params[LearningParams.hyper_parameters][HyperParams.error_type],
                                 EmbStas.runtime: rtime,
                                 EmbStas.learning_rate: self.hyper_params[EmbStas.learning_rate],
                                 EmbStas.epochs: self.hyper_params[EmbStas.epochs],
                                 EmbStas.optimizer: self.hyper_params[EmbStas.optimizer],
                                 EmbStas.optimizer_params: optimizer_params,
                                 EmbStas.eigenvector_method: self.learning_params[LearningParams.eigenvector_method],
                                 EmbStas.pi_max_err: self.hyper_params[EmbStas.pi_max_err],
                                 EmbStas.device: self.learning_params[LearningParams.device],
                                 EmbStas.dtype: self.learning_params[LearningParams.dtype]
                                 }
        self.emb_params_statistics = params_statistic_dict

    def prepare_stuck_params_statistics(self, centrality, adj_matrix, learning_target, learning_params, err_msg,
                                        optimizer_params):
        stuck_params_statistic_dict = {EmbStas.id: datetime.datetime.now(),
                                       EmbStas.centrality: centrality,
                                       RbcMatrices.adjacency_matrix: adj_matrix,
                                       EmbStas.target: learning_target,
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

