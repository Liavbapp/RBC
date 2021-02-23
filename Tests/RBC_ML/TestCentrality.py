import datetime
import random
import torch
import networkx as nx
from Utils.CommonStr import HyperParams
from Utils.CommonStr import RbcMatrices
from Tests.Tools import saver
from Utils.CommonStr import StatisticsParams as Stas
from Components.RBC_ML.RbcML import learn_models
from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import LearningParams
from Utils.CommonStr import OptimizerTypes
from Utils.CommonStr import ErrorTypes
from Utils.CommonStr import EigenvectorMethod
from Utils.CommonStr import TorchDevice
from Utils.CommonStr import TorchDtype
from Utils.CommonStr import Centralities


def get_params(g, centrality):
    device = TorchDevice.cpu
    dtype = TorchDtype.float
    adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=dtype)
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    learning_target = get_learning_target(centrality, g, nodes_mapping, device, dtype)

    hyper_params = {HyperParams.learning_rate: 1e-3,
                    HyperParams.epochs: 700,
                    HyperParams.momentum: 0,
                    HyperParams.optimizer: OptimizerTypes.RmsProp,
                    HyperParams.pi_max_err: 0.0001,
                    HyperParams.error_type: ErrorTypes.mse
                    }
    learning_params = {LearningParams.hyper_parameters: hyper_params,
                       LearningParams.adjacency_matrix: adj_matrix,
                       LearningParams.target: learning_target,
                       LearningParams.src_src_one: True,
                       LearningParams.src_row_zeros: False,
                       LearningParams.target_col_zeros: False,
                       LearningParams.sigmoid: True,
                       LearningParams.eigenvector_method: EigenvectorMethod.power_iteration,
                       LearningParams.device: device,
                       LearningParams.dtype: dtype
                       }

    return hyper_params, learning_params


def save_statistics(learning_params, hyper_params, t_model, r_model, final_error, rtime, rbc_pred, centrality):
    kwargs_dict = {RbcMatrices.adjacency_matrix: learning_params[LearningParams.adjacency_matrix],
                   RbcMatrices.routing_policy: r_model,
                   RbcMatrices.traffic_matrix: t_model,
                   Stas.centrality: centrality,
                   Stas.target: learning_params[LearningParams.target],
                   Stas.prediction: rbc_pred,
                   Stas.error: final_error,
                   Stas.error_type: learning_params[LearningParams.hyper_parameters][HyperParams.error_type],
                   Stas.sigmoid: learning_params[LearningParams.sigmoid],
                   Stas.src_src_one: learning_params[LearningParams.src_src_one],
                   Stas.src_row_zeros: learning_params[LearningParams.src_row_zeros],
                   Stas.target_col_zeros: learning_params[LearningParams.target_col_zeros],
                   Stas.runtime: rtime,
                   Stas.learning_rate: hyper_params[Stas.learning_rate],
                   Stas.epochs: hyper_params[Stas.epochs],
                   Stas.momentum: hyper_params[Stas.momentum],
                   Stas.optimizer: hyper_params[Stas.optimizer],
                   Stas.pi_max_err: hyper_params[Stas.pi_max_err],
                   Stas.eigenvector_method: learning_params[LearningParams.eigenvector_method],
                   Stas.device: learning_params[LearningParams.device],
                   Stas.dtype: learning_params[LearningParams.dtype]
                   }

    saver.save_info(**kwargs_dict)


def get_rbc_handler(learning_params, hyper_params):
    eigenvector_method = learning_params[LearningParams.eigenvector_method]
    pi_max_error = hyper_params[HyperParams.pi_max_err]
    device = learning_params[LearningParams.device]
    dtype = learning_params[LearningParams.dtype]
    rbc_handler = RBC(eigenvector_method, pi_max_error, device, dtype)

    return rbc_handler


def get_learning_target(centrality, g, nodes_mapping, device, dtype):
    if centrality == Centralities.SPBC:
        tensor_raw = torch.tensor(list(nx.betweenness_centrality(g).values()), dtype=dtype, device=device)
        tensor_norm = tensor_raw.clone()
        for node_val, node_idx in nodes_mapping.items():
            tensor_norm[node_val] = tensor_raw[node_idx]
        return tensor_norm
    else:
        raise NotImplementedError

def test_centrality_on_graph(centrality, g, test_num):
    print(f' Testing {centrality} Centrality - test number {test_num}')
    hyper_params, learning_params = get_params(g, centrality)
    rbc_handler = get_rbc_handler(learning_params, hyper_params)
    adj_matrix = learning_params[LearningParams.adjacency_matrix]
    learning_target = learning_params[LearningParams.target]

    start_time = datetime.datetime.now()
    try:
        t_model, r_model, final_error = learn_models(learning_params)
        runtime = datetime.datetime.now() - start_time
        rbc_pred = rbc_handler.compute_rbc(g, r_model, t_model)
        print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
        save_statistics(learning_params, hyper_params, t_model, r_model, final_error, runtime, rbc_pred, centrality)
    except Exception as e:
        saver.save_info_stuck(centrality, adj_matrix, learning_target, learning_params, str(e))


def generate_10_rand_graphs():
    graphs = []
    for i in range(4, 14):
        g = nx.complete_graph(i)
        edge_lst = list(nx.edges(g))
        rand_edges = random.sample(edge_lst, int(0.8 * len(edge_lst)))
        h = nx.Graph()
        h.add_nodes_from(g)
        h.add_edges_from(rand_edges)
        graphs.append(h)
    return graphs


def test_centrality(centrality):
    tested_graphs = generate_10_rand_graphs()
    for i in range(1, len(tested_graphs)):
        test_centrality_on_graph(centrality, tested_graphs[i], i)


if __name__ == '__main__':
    test_centrality(Centralities.SPBC)
