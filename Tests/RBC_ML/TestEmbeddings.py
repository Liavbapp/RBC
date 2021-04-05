import datetime
import os
import random
import sys
import torch
import numpy as np
import networkx as nx
from Utils import Paths

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
    train_data, validation_data, test_data = split_to_train_validation_test(params_man)
    embed_train, embed_validation, embed_test = generate_embeddings(train_data, validation_data, test_data, params_man)

    trained_model, train_time, train_err = train_model(nn_model=nn_model, train_data=train_data,
                                                       p_man=params_man, optimizer=optimizer,
                                                       embeddings_train=embed_train,
                                                       validation_data=validation_data,
                                                       embeddings_validation=embed_validation)
    train_example = (train_data['Rs'][0], train_data['Ts'][0], train_data['Gs'][0])
    test_example = (test_data['Rs'][0], test_data['Ts'][0], test_data['Gs'][0])
    expected_rbc, actual_rbc, test_routing_policy = test_model(model=trained_model, train_example=train_example,
                                                               test_example=test_example, p_man=params_man,
                                                               test_embeddings=embed_test[0])
    params_man.train_path_params = train_data['path_params']
    params_man.test_path_params = test_data['path_params']
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


def generate_seeds(train_data, validation_data, test_data):
    train_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in train_data['path_params']]
    validation_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in
                        validation_data['path_params']]
    test_seeds = [list(param_info.values())[0][EmbeddingPathParams.seed] for param_info in test_data['path_params']]
    return train_seeds, validation_seeds, test_seeds


def split_to_train_validation_test(p_man):
    graphs_paths = get_graphs_path()
    Rs, Ts, Gs, path_params = extract_info_from_path(graphs_paths, p_man)
    train_data_lists = {'Rs': Rs[2:], 'Ts': Ts[2:], 'Gs': Gs[2:], 'path_params': path_params[2:]}
    validation_data_lists = {'Rs': [Rs[1]], 'Ts': [Ts[1]], 'Gs': [Gs[1]], 'path_params': [path_params[1]]}
    test_data_lists = {'Rs': [Rs[0]], 'Ts': [Ts[0]], 'Gs': [Gs[0]], 'path_params': [path_params[0]]}
    return train_data_lists, validation_data_lists, test_data_lists


def generate_embeddings(train_data, validation_data, test_data, params_man):
    combined_embed = get_combined_embeddings(train_data, validation_data, test_data, params_man)
    embed_train, embed_validation, embed_test = split_embeddings(combined_embed,
                                                                 len(train_data['Rs']),
                                                                 len(validation_data['Rs']),
                                                                 len(test_data['Rs']))
    return embed_train, embed_validation, embed_test


def get_combined_embeddings(train_data, validation_data, test_data, params_man):
    train_seeds, validation_seeds, test_seeds = generate_seeds(train_data, validation_data, test_data)
    train_graphs = train_data['Gs']
    validation_graphs = validation_data['Gs']
    test_graphs = test_data['Gs']
    all_graphs = train_graphs + validation_graphs + test_graphs
    all_seeds = train_seeds + validation_seeds + test_seeds
    preprocessor = PreProcessor(dim=params_man.embedding_dimensions, device=params_man.device, dtype=params_man.dtype)
    embeddings = preprocessor.compute_embeddings(all_graphs, all_seeds)
    return embeddings


def split_embeddings(embeddings, train_len, validation_len, test_len):
    embeddings_train = embeddings[0: train_len]
    embeddings_validation = embeddings[train_len: train_len + validation_len]
    embeddings_test = embeddings[train_len + validation_len: train_len + validation_len + test_len]
    return embeddings_train, embeddings_validation, embeddings_test


def get_graphs_path(num_nodes=4):
    lst = Paths.train_paths_7_nodes
    # random.shuffle(lst)
    return lst


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


def train_model(nn_model, train_data, validation_data, p_man, optimizer, embeddings_train, embeddings_validation):
    device = p_man.device
    dtype = p_man.dtype
    dim = p_man.embedding_dimensions
    preprocessor = PreProcessor(dim=dim, device=device, dtype=dtype)
    samples_train = preprocessor.generate_all_samples(embeddings=embeddings_train, Rs=train_data['Rs'])
    samples_validation = preprocessor.generate_all_samples(embeddings=embeddings_validation, Rs=validation_data['Rs'])
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
        EmbStat.embd_dim: 32,
        'seed_range': 100,
        EmbStat.csv_save_path: csv_save_path,
        EmbeddingOutputs.root_path: embedding_outputs_root_path,
        HyperParams.optimizer: OptimizerTypes.Adam,
        HyperParams.learning_rate: 1e-4,
        HyperParams.epochs: 1000,
        HyperParams.batch_size: 1024,
        HyperParams.weight_decay: 0,
        HyperParams.momentum: 0,
        HyperParams.pi_max_err: 0.00001,
        HyperParams.error_type: ErrorTypes.mse,
        EmbStat.eigenvector_method: EigenvectorMethod.power_iteration
    }
    arr = [params_statistics1]

    for i in range(0, len(arr)):
        print(f'{i} out of {len(arr)}')
        run_test(arr[i])
