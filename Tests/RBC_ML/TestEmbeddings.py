import datetime
import os
import random
import sys

import torch
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.NeuralNetwork import NeuralNetwork as EmbeddingNeuralNetwork
from Components.Embedding import EmbeddingML
from Components.Embedding.PreProcessor import PreProcessor
from Components.RBC_ML.Optimizer import Optimizer
from Components.RBC_REG.RBC import RBC
from Tests.RBC_ML.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import EigenvectorMethod, EmbeddingStatistics as EmbStat, Centralities, TorchDevice, TorchDtype, \
    HyperParams, OptimizerTypes, ErrorTypes, EmbeddingOutputs, EmbeddingPathParams


def run_test(pr_st):
    nn_model = init_nn_model(pr_st)
    optimizer = Optimizer(model=nn_model, name=pr_st[EmbStat.optimizer], learning_rate=pr_st[EmbStat.learning_rate],
                          momentum=pr_st[HyperParams.momentum], weight_decay=pr_st[HyperParams.weight_decay])
    params_man = EmbeddingsParams(pr_st)
    train_data, validation_data, tests_data = split_to_train_validation_test(params_man)

    train_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in train_data[3]]
    validation_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in validation_data[3]]
    test_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in tests_data[3]]

    embeddings_train, embeddings_validation, embeddings_test = generate_embeddings(train_graphs=train_data[2],
                                                                                   validation_graphs=validation_data[2],
                                                                                   test_graphs=tests_data[2],
                                                                                   train_seeds=train_seeds,
                                                                                   validation_seeds=validation_seeds,
                                                                                   test_seeds=test_seeds,
                                                                                   prm_st=pr_st)
    trained_model, train_time, train_err = train_model(nn_model=nn_model, train_info=train_data, p_man=params_man,
                                                       optimizer=optimizer, embeddings_train=embeddings_train,
                                                       validation_info=validation_data,
                                                       embeddings_validation=embeddings_validation)
    train_example = (train_data[0][0], train_data[1][0], train_data[2][0])
    test_example = (tests_data[0][0], tests_data[1][0], tests_data[2][0])
    expected_rbc, actual_rbc, test_routing_policy = test_model(model=trained_model, train_example=train_example,
                                                               test_example=test_example, p_man=params_man,
                                                               test_embeddings=embeddings_test[0])
    params_man.train_path_params = train_data[3]
    params_man.test_path_params = tests_data[3]
    params_man.trained_model = trained_model
    params_man.train_runtime = train_time
    params_man.train_error = train_err
    params_man.expected_rbc = expected_rbc
    params_man.actual_rbc = actual_rbc
    params_man.test_routing_policy = test_routing_policy
    params_man.test_graph = test_example[2]
    params_man.network_structure = nn_model.linear_relu_stack.__str__()
    params_man.optimizer_params = optimizer.get_optimizer_params()
    params_man.prepare_params_statistics()
    params_man.save_params_statistics()


def split_to_train_validation_test(p_man):
    graphs_paths = get_graphs_path()
    Rs, Ts, Gs, path_params = extract_info_from_path(graphs_paths, p_man)
    train_data_lists = Rs[2:], Ts[2:], Gs[2:], path_params[2:]
    validation_data_lists = [Rs[1]], [Ts[1]], [Gs[1]], [path_params[1]]
    test_data_lists = [Rs[0]], [Ts[0]], [Gs[0]], [path_params[0]]
    return train_data_lists, validation_data_lists, test_data_lists


def generate_embeddings(train_graphs, test_graphs, train_seeds, test_seeds, prm_st, validation_seeds,
                        validation_graphs):
    train_g_nbr = len(train_graphs)
    validation_g_nbr = len(validation_graphs)
    test_g_nbr = len(test_graphs)
    all_graphs = train_graphs + validation_graphs + test_graphs
    all_seeds = train_seeds + validation_seeds + test_seeds
    preprocessor = PreProcessor(dim=prm_st[EmbStat.embd_dim], device=prm_st[EmbStat.device],
                                dtype=prm_st[EmbStat.dtype])
    embeddings = preprocessor.compute_embeddings(all_graphs, all_seeds)
    embeddings_train = embeddings[0: train_g_nbr]
    embeddings_validation = embeddings[train_g_nbr: train_g_nbr + validation_g_nbr]
    embeddings_test = embeddings[train_g_nbr + validation_g_nbr: train_g_nbr + validation_g_nbr + test_g_nbr]
    return embeddings_train, embeddings_validation, embeddings_test


def get_graphs_path():
    train_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\390',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\391',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\392',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\393',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\394',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\395',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\396',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\397',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\398',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\399',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\400',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\401',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\402',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\403',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\404']

    # train_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\15',
    #                r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\9',
    #                r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\10',
    #                r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\12']
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\13',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\14']
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\16',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\17',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\18',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\19',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\20',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\21',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\22']

    return train_paths


def extract_info_from_path(paths, p_man):
    R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=p_man.dtype, device=p_man.device) for path
             in paths]
    T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=p_man.dtype, device=p_man.device) for path in
             paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in paths]
    seeds = random.sample(range(0, p_man.seed_range), len(paths))
    path_params = [{path: {EmbeddingPathParams.seed: seed}} for path, seed in zip(paths, seeds)]
    return R_lst, T_lst, G_lst, path_params


def init_nn_model(param_embed):
    device = param_embed[EmbStat.device]
    dtype = param_embed[EmbStat.dtype]
    embed_dimension = param_embed[EmbStat.embd_dim]
    model = EmbeddingNeuralNetwork(embed_dimension, device, dtype)
    return model


def train_model(nn_model, train_info, validation_info, p_man, optimizer, embeddings_train, embeddings_validation):
    train_Rs, train_Ts, train_Gs, train_seeds = train_info
    validation_Rs, validation_Ts, validation_Gs, validation_seeds = validation_info
    device = p_man.device
    dtype = p_man.dtype
    dim = p_man.embedding_dimensions
    preprocessor = PreProcessor(dim=dim, device=device, dtype=dtype)
    samples_train = preprocessor.generate_all_samples(embeddings=embeddings_train, Rs=train_Rs)
    samples_validation = preprocessor.generate_all_samples(embeddings=embeddings_validation, Rs=validation_Rs)
    start_time = datetime.datetime.now()
    model_trained, train_error = EmbeddingML.train_model(nn_model, samples_train, samples_validation, p_man, optimizer)
    train_time = datetime.datetime.now() - start_time
    print(f'train time: {train_time}')
    return model_trained, train_time, train_error


def test_model(model, train_example, test_example, p_man: EmbeddingsParams, test_embeddings):
    pi_max_err = p_man.hyper_params[HyperParams.pi_max_err]
    device = p_man.device
    dtype = p_man.dtype

    test_R, test_T, test_G = test_example
    train_R, train_T, train_G = train_example

    rbc_train = RBC(EigenvectorMethod.power_iteration, pi_max_error=pi_max_err, device=device, dtype=dtype)
    rbc_test = RBC(EigenvectorMethod.power_iteration, pi_max_error=pi_max_err, device=device, dtype=dtype)

    expected_rbc = rbc_train.compute_rbc(train_G, train_R, train_T)
    test_r_policy = EmbeddingML.predict_routing(model, test_embeddings, p_man)
    actual_rbc = rbc_test.compute_rbc(test_G, test_r_policy, test_T)

    print(f'expected rbc: {expected_rbc}')
    print(f'actual rbc: {actual_rbc}')

    return expected_rbc, actual_rbc, test_r_policy


if __name__ == '__main__':
    csv_save_path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\Combined_Results\\With_Embedding\\statistics_embedding.csv'
    embedding_outputs_root_path = f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\Embedding_results'

    params_statistics1 = {
        EmbStat.centrality: Centralities.SPBC,
        EmbStat.device: TorchDevice.gpu,
        EmbStat.dtype: TorchDtype.float,
        EmbStat.embd_dim: 5,
        'seed_range': 100,
        EmbStat.csv_save_path: csv_save_path,
        EmbeddingOutputs.root_path: embedding_outputs_root_path,
        HyperParams.optimizer: OptimizerTypes.Adam,
        HyperParams.learning_rate: 1e-4,
        HyperParams.epochs: 1200,
        HyperParams.batch_size: 1024,
        HyperParams.weight_decay: 0.01,
        HyperParams.momentum: 0,
        HyperParams.pi_max_err: 0.00001,
        HyperParams.error_type: ErrorTypes.mse,
        EmbStat.eigenvector_method: EigenvectorMethod.power_iteration
    }
    arr = [params_statistics1]

    for i in range(0, len(arr)):
        print(f'{i} out of {len(arr)}')
        run_test(arr[i])
