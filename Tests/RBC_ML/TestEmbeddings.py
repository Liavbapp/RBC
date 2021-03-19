import datetime

import torch
import numpy as np
import networkx as nx
from Components.Embedding.NeuralNetwork import NeuralNetwork as EmbeddingNeuralNetwork
from Components.Embedding.EmbeddingML import EmbeddingML
from Components.Embedding.PreProcessor import PreProcessor
from Components.RBC_ML.Optimizer import Optimizer
from Components.RBC_REG.RBC import RBC
from Tests.RBC_ML.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import EigenvectorMethod, EmbeddingStatistics as EmbedStats, Centralities, TorchDevice, TorchDtype, \
    HyperParams, OptimizerTypes, ErrorTypes


def get_paths():
    train_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\8',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\7',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\9',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\10',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\11',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\12',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\13',
                   # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\15',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\16']
    test_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14']
    return train_paths, test_paths


def extract_info_from_path(paths):
    R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in paths]

    return R_lst, T_lst, G_lst


def init_nn_model(param_embed):
    device = param_embed[EmbedStats.device]
    dtype = param_embed[EmbedStats.dtype]
    embed_dimension = param_embed[EmbedStats.embedding_dimensions]
    model = EmbeddingNeuralNetwork(embed_dimension, device, dtype)
    return model


def train_model(nn_model, train_info, p_man, optimizer):
    train_Rs, train_Ts, train_Gs = train_info
    train_pre_processor = PreProcessor(dimensions=p_man.embedding_dimensions, Gs=train_Gs, Rs=train_Rs, Ts=train_Ts)
    samples = train_pre_processor.pre_process_data()
    start_time = datetime.datetime.now()
    model_trained = EmbeddingML.train_model(nn_model, samples, p_man, optimizer)
    train_time = datetime.datetime.now() - start_time
    print(f'train time: {train_time}')
    return model_trained, train_time


def test_model(model, test_data, train_example, p_man: EmbeddingsParams):
    pi_max_err = p_man.hyper_params[HyperParams.pi_max_err]
    device = p_man.device
    dtype = p_man.dtype

    test_R, test_T, test_G = test_data
    train_R, train_T, train_G = train_example

    test_preprocessor = PreProcessor(p_man.embedding_dimensions, [test_G], [test_R], [test_T])
    test_embeddings = test_preprocessor.compute_embeddings()[0]
    rbc_train = RBC(EigenvectorMethod.power_iteration, pi_max_error=pi_max_err, device=device, dtype=dtype)
    rbc_test = RBC(EigenvectorMethod.power_iteration, pi_max_error=pi_max_err, device=device, dtype=dtype)

    expected_rbc = rbc_train.compute_rbc(train_G, train_R, train_T)
    actual_rbc = rbc_test.compute_rbc(test_G, EmbeddingML.predict_routing(model, test_embeddings), test_T)

    print(f'expected rbc: {train_example}')
    print(f'actual rbc: {actual_rbc}')

    return expected_rbc, actual_rbc


if __name__ == '__main__':
    train_paths, tests_paths = get_paths()
    train_data = extract_info_from_path(train_paths)
    test_data = extract_info_from_path(tests_paths)

    params_statistics = {
        EmbedStats.centrality: Centralities.SPBC,
        EmbedStats.device: TorchDevice.gpu,
        EmbedStats.dtype: TorchDtype.float,
        EmbedStats.embedding_dimensions: 5,
        HyperParams.optimizer: OptimizerTypes.AdaMax,
        HyperParams.learning_rate: 1e-4,
        HyperParams.momentum: 0,
        HyperParams.epochs: 2000,
        HyperParams.batch_size: 512,
        HyperParams.pi_max_err: 0.000001,
        HyperParams.error_type: ErrorTypes.mse,
        EmbedStats.eigenvector_method: EigenvectorMethod.power_iteration
    }
    nn_model = init_nn_model(params_statistics)
    params_statistics[EmbedStats.network_structure] = nn_model.linear_relu_stack.__str__()

    optimizer_name = params_statistics[EmbedStats.optimizer]
    learning_rate = params_statistics[EmbedStats.learning_rate]
    momentum = params_statistics[HyperParams.momentum]
    optimizer = Optimizer(nn_model, optimizer_name, learning_rate, momentum=momentum)
    params_statistics[EmbedStats.optimizer_params] = optimizer.get_optimizer_params()

    params_man = EmbeddingsParams(params_statistics)
    trained_model, train_time = train_model(nn_model, train_data, params_man, optimizer)
    params_man.trained_model = trained_model
    params_man.train_runtime = train_time

    train_example = (train_data[0][0], train_data[1][0], train_data[2][0])
    test_model(trained_model, train_data, train_example, params_man)
