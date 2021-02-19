import datetime

import torch
import networkx as nx
from Utils.CommonStr import HyperParams
from Utils.CommonStr import RbcMatrices
from Tests.Tools import saver
from Utils.CommonStr import StatisticsParams as Stas
from Components.RBC_ML.RbcML import learn_models
import Components.RBC_REG.RBC as RBC_REG
from Utils.CommonStr import LearningParams
from Utils.CommonStr import OptimizerTypes
from Utils.CommonStr import ErrorTypes
DEVICE = torch.device('cuda:0')
DTYPE = torch.float


def test_spbc_on_graph(g, adj_matrix, test_num):
    print(f'started testing figure_{test_num}')
    nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    betweenness_tensor = get_betweenness_tensor(g, nodes_mapping)

    hyper_params = {HyperParams.learning_rate: 1e-3,
                    HyperParams.epochs: 3000,
                    HyperParams.momentum: 0.00,
                    HyperParams.optimizer: OptimizerTypes.adam,
                    HyperParams.pi_max_err: 0.0001,
                    HyperParams.error_type: ErrorTypes.mse
                    }
    learning_params = {LearningParams.hyper_parameters: hyper_params,
                       LearningParams.adjacency_matrix: adj_matrix,
                       LearningParams.target: betweenness_tensor,
                       LearningParams.src_src_one: True,
                       LearningParams.src_row_zeros: False,
                       LearningParams.target_col_zeros: False,
                       LearningParams.sigmoid: True
                       }

    start_time = datetime.datetime.now()
    t_model, r_model, final_error = learn_models(learning_params)
    rtime = datetime.datetime.now() - start_time
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC_REG.compute_rbc(g, r_model, t_model)
    print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
    kwargs_dict = {RbcMatrices.adjacency_matrix: adj_matrix,
                   RbcMatrices.routing_policy: r_model,
                   RbcMatrices.traffic_matrix: t_model,
                   Stas.centrality: 'SPBC',
                   Stas.target: betweenness_tensor,
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
                   Stas.pi_max_err: hyper_params[Stas.pi_max_err]
                   }
    saver.save_info(**kwargs_dict)



def get_betweenness_tensor(g, nodes_mapping):
    tensor_raw = torch.tensor(list(nx.betweenness_centrality(g, endpoints=True).values()), dtype=DTYPE, device=DEVICE)
    tensor_norm = tensor_raw.clone()
    for node_val, node_idx in nodes_mapping.items():
        tensor_norm[node_val] = tensor_raw[node_idx]

    return tensor_norm


def test_spbc():
    edge_lst_0 = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edge_lst_1 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)]
    edge_lst_2 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5)]
    edge_lst_3 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    edge_lst_4 = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 5)]
    edge_lst_5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
    edge_lst_6 = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4), (0, 4), (4, 5)]
    edge_lst_7 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4)]
    edge_lst_8 = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4), (5, 7),
                  (6, 7), (4, 8), (7, 8), (8, 9), (4, 9)]
    all_edge_lst = [edge_lst_0, edge_lst_1, edge_lst_2, edge_lst_3, edge_lst_4, edge_lst_5, edge_lst_6, edge_lst_7,
                    edge_lst_8]

    i = 0
    for edge_lst in all_edge_lst:
        g = nx.Graph(edge_lst)
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=DTYPE)
        test_spbc_on_graph(g, adj_matrix, test_num=i)
        i += 1


if __name__ == '__main__':
    test_spbc()
