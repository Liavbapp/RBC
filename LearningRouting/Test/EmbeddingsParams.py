import datetime

from Utils import Saver
from Utils.CommonStr import  HyperParams, LearningParams, OptimizerTypes, EmbeddingStatistics, EmbeddingOutputs


def get_centrality_params(centrality, device, dtype):
    # TODO: get centrality params from csv
    return ""


class EmbeddingsParams:
    def __init__(self, params_dict):
        self.path_obj = params_dict['path_obj']
        self.num_nodes = params_dict['num_nodes']
        self.technique = params_dict['technique']

        self.n_routing_policy_per_graph = self.path_obj.n_routing_per_graph
        self.graphs_root_path = self.path_obj.root_path
        self.graphs_desc = self.path_obj.description

        self.seeds_per_train_graph = params_dict[EmbeddingStatistics.n_seeds_train_graph]
        self.n_rand_samples_graph = params_dict[EmbeddingStatistics.n_random_samples_per_graph]
        self.centrality = params_dict[EmbeddingStatistics.centrality]
        self.csv_path = params_dict[EmbeddingStatistics.csv_save_path]
        self.trained_model_path = params_dict[EmbeddingOutputs.trained_model_root_path]
        self.embedding_dimensions = params_dict[EmbeddingStatistics.embd_dim]
        self.device = params_dict[EmbeddingStatistics.device]
        self.dtype = params_dict[EmbeddingStatistics.dtype]
        self.seed_range = params_dict['seed_range']
        self.embedding_alg_name = params_dict[EmbeddingStatistics.embedding_alg]
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
                                LearningParams.eigenvector_method: params_dict[EmbeddingStatistics.eigenvector_method],
                                LearningParams.device: self.device,
                                LearningParams.dtype: self.dtype
                                }

        self.n_graphs_train = None
        self.n_graphs_validation = None
        self.n_graphs_test = None
        self.optimizer_params = None
        self.trained_model = None
        self.train_runtime = None
        self.train_error = None
        self.validation_error = None
        self.expected_rbcs = None
        self.actual_rbcs = None
        self.emb_params_statistics = None
        self.emb_params_stuck_statics = None
        self.network_structure = None
        self.euclidean_dis_median = None
        self.kendall_tau_avg = None
        self.pearson_avg = None
        self.spearman_avg = None

    def prepare_params_statistics(self):

        params_statistic_dict = {
                                 EmbeddingOutputs.graphs_root_path: self.graphs_root_path,
                                 EmbeddingOutputs.trained_model_root_path: self.trained_model_path,
                                 EmbeddingStatistics.csv_save_path: self.csv_path,
                                 EmbeddingStatistics.id: datetime.datetime.now(),
                                 EmbeddingStatistics.centrality: self.centrality,
                                 EmbeddingStatistics.centrality_params: self.learning_params[LearningParams.centrality_params],
                                 EmbeddingStatistics.n_graphs_train: self.n_graphs_train,
                                 EmbeddingStatistics.n_graphs_validation: self.n_graphs_validation,
                                 EmbeddingStatistics.n_graphs_test: self.n_graphs_test,
                                 EmbeddingStatistics.n_seeds_train_graph: self.seeds_per_train_graph,
                                 EmbeddingStatistics.n_routing_policy_per_graph: self.n_routing_policy_per_graph,
                                 EmbeddingStatistics.n_random_samples_per_graph: self.n_rand_samples_graph,
                                 EmbeddingStatistics.graphs_desc: self.graphs_desc,
                                 EmbeddingStatistics.embedding_alg: self.embedding_alg_name,
                                 EmbeddingStatistics.embd_dim: self.embedding_dimensions,
                                 EmbeddingStatistics.rbcs_expected: self.expected_rbcs,
                                 EmbeddingStatistics.rbcs_actual: self.actual_rbcs,
                                 EmbeddingStatistics.train_error: self.train_error,
                                 EmbeddingStatistics.validation_error: self.validation_error,
                                 EmbeddingStatistics.error_type: self.learning_params[LearningParams.hyper_parameters][HyperParams.error_type],
                                 EmbeddingStatistics.euclidean_distance_median: self.euclidean_dis_median,
                                 EmbeddingStatistics.kendall_tau_b_avg: self.kendall_tau_avg,
                                 EmbeddingStatistics.pearson_avg: self.pearson_avg,
                                 EmbeddingStatistics.spearman_avg: self.spearman_avg,
                                 EmbeddingStatistics.train_runtime: self.train_runtime,
                                 EmbeddingStatistics.network_structure: self.network_structure,
                                 EmbeddingStatistics.learning_rate: self.hyper_params[EmbeddingStatistics.learning_rate],
                                 EmbeddingStatistics.epochs: self.hyper_params[EmbeddingStatistics.epochs],
                                 EmbeddingStatistics.batch_size: self.hyper_params[EmbeddingStatistics.batch_size],
                                 EmbeddingStatistics.weight_decay: self.hyper_params[EmbeddingStatistics.weight_decay],
                                 EmbeddingStatistics.optimizer: self.hyper_params[EmbeddingStatistics.optimizer],
                                 EmbeddingStatistics.optimizer_params: self.optimizer_params,
                                 EmbeddingStatistics.eigenvector_method: self.learning_params[LearningParams.eigenvector_method],
                                 EmbeddingStatistics.pi_max_err: self.hyper_params[EmbeddingStatistics.pi_max_err],
                                 EmbeddingStatistics.device: self.learning_params[LearningParams.device],
                                 EmbeddingStatistics.dtype: self.learning_params[LearningParams.dtype],
                                 EmbeddingOutputs.trained_model: self.trained_model,
                                 }
        self.emb_params_statistics = params_statistic_dict

    def prepare_stuck_params_statistics(self, centrality, learning_target, learning_params, err_msg,
                                        optimizer_params):
        stuck_params_statistic_dict = {
            EmbeddingStatistics.id: datetime.datetime.now(),
            EmbeddingStatistics.csv_save_path: self.csv_path,
            EmbeddingStatistics.centrality: centrality,
            EmbeddingStatistics.rbcs_expected: learning_target,
            LearningParams.name: learning_params,
            EmbeddingStatistics.comments: err_msg,
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
