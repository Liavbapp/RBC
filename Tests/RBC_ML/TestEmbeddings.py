import datetime
import os
import random
import sys
import torch
import numpy as np
import networkx as nx
import scipy.stats
import scipy.spatial.distance
from Utils import Paths
import itertools
from Utils.EmbeddingAlg import get_embedding_algo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.NeuralNetwork import NeuralNetwork as EmbeddingNeuralNetwork, \
    NeuralNetworkNodesEmbeddingRouting, \
    NeuralNetworkGraphEmbeddingRouting, NeuralNetworkGraphEmbeddingRbc, NeuralNetworkNodeEmbeddingSourceTargetRouting
from Components.Embedding import EmbeddingML
from Components.Embedding.PreProcessor import PreProcessor
from Utils.Optimizer import Optimizer
from Components.RBC_REG.RBC import RBC
from Tests.RBC_ML.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import EigenvectorMethod, EmbeddingStatistics as EmbStat, Centralities, TorchDevice, TorchDtype, \
    HyperParams, OptimizerTypes, ErrorTypes, EmbeddingOutputs, EmbeddingPathParams, EmbeddingAlgorithms, Techniques, \
    NumRandomSamples


def run_test(pr_st):
    """
    The main flow of the testing
    :param pr_st: dictionary with all parameters required for the current run
    :return: generating param_man object with all results of the current run and saving it to csv
    """

    # init
    nn_model = init_nn_model(pr_st)
    optimizer = init_optimizer(pr_st, nn_model)
    params_man = EmbeddingsParams(pr_st)

    # data handling
    split_data_res = split_to_train_validation_test(params_man)
    train_data, validation_data, test_data = split_data_res
    embed_train, embed_validation, embed_test = generate_embeddings(train_data, validation_data, test_data, params_man)

    # training
    train_res = train_model(nn_model, optimizer, params_man, train_data, embed_train, validation_data, embed_validation)
    trained_model, train_err, validation_err, train_time = train_res

    # testing
    test_res = test_model(trained_model, params_man, test_data, embed_test)
    # expected_rbcs, actual_rbcs, euclidean_dist_median, kendall_tau_avg, spearman_avg, pearsonr_avg = test_res

    # update param manager object
    update_params_man(params_man, split_data_res, train_res, test_res, optimizer)

    # save results to csv
    params_man.prepare_params_statistics()
    params_man.save_params_statistics()


def init_optimizer(pr_st, nn_model):
    optimizer_name, lr, momentum = pr_st[EmbStat.optimizer], pr_st[EmbStat.learning_rate], pr_st[HyperParams.momentum]
    wd = pr_st[HyperParams.weight_decay]
    optimizer = Optimizer(model=nn_model, name=optimizer_name, learning_rate=lr, momentum=momentum, weight_decay=wd)

    return optimizer


def split_to_train_validation_test(p_man):
    train_paths, validation_paths, test_paths = get_tvt_paths(p_man)

    Rs_train, Ts_train, Gs_train = load_rtg_from_paths(train_paths, p_man)
    Rs_val, Ts_val, Gs_val = load_rtg_from_paths(validation_paths, p_man)
    Rs_test, Ts_test, Gs_test = load_rtg_from_paths(test_paths, p_man)

    len_train, len_val, len_test = len(train_paths), len(validation_paths), len(test_paths)
    train_seeds, validation_seeds, test_seeds = get_seeds(len_train, len_val, len_test, p_man.seed_range)

    train_data_lists = {'Rs': Rs_train, 'Ts': Ts_train, 'Gs': Gs_train, 'seeds': train_seeds}
    validation_data_lists = {'Rs': Rs_val, 'Ts': Ts_val, 'Gs': Gs_val, 'seeds': validation_seeds}
    test_data_lists = {'Rs': Rs_test, 'Ts': Ts_test, 'Gs': Gs_test, 'seeds': test_seeds}

    return train_data_lists, validation_data_lists, test_data_lists


def get_tvt_paths(p_man):
    train_paths = [path[0] for path in list(os.walk(path_obj.train_graphs_path))[1:]] * p_man.seeds_per_train_graph
    validation_paths = [path[0] for path in list(os.walk(path_obj.validation_graphs_path))[1:]]
    test_paths = [path[0] for path in list(os.walk(path_obj.test_graphs_path))[1:]]
    return train_paths, validation_paths, test_paths


def load_rtg_from_paths(paths, p_man):
    routing_np, traffic_np, adj_np = 'routing_policy.npy', 'traffic_mat.npy', 'adj_mat.npy'
    Rs = [torch.tensor(np.load(f'{path}\\{routing_np}'), dtype=p_man.dtype, device=p_man.device) for path in paths]
    Ts = [torch.tensor(np.load(f'{path}\\{traffic_np}'), dtype=p_man.dtype, device=p_man.device) for path in paths]
    Gs = [nx.convert_matrix.from_numpy_matrix(np.load(f'{path}\\{adj_np}')) for path in paths]
    return Rs, Ts, Gs


def get_seeds(len_train_paths, len_val_paths, len_test_paths, seed_range):
    len_all_paths = len_train_paths + len_val_paths + len_test_paths
    seeds = random.sample(range(0, seed_range), len_all_paths)

    train_seeds = seeds[0:len_train_paths]
    val_seeds = seeds[len_train_paths: len_train_paths + len_val_paths]
    test_seeds = seeds[len_train_paths + len_val_paths: len_all_paths]

    return train_seeds, val_seeds, test_seeds


def generate_embeddings(train_data, validation_data, test_data, params_man):
    train_rs_len, val_rs_len, test_rs_len = len(train_data['Rs']), len(validation_data['Rs']), len(test_data['Rs'])
    combined_embed = get_combined_embeddings(train_data, validation_data, test_data, params_man)
    embed_train, embed_validation, embed_test = split_embeddings(combined_embed, train_rs_len, val_rs_len, test_rs_len)

    return embed_train, embed_validation, embed_test


def get_combined_embeddings(train_data, validation_data, test_data, params_man):
    train_seeds, validation_seeds, test_seeds = train_data['seeds'], validation_data['seeds'], test_data['seeds']
    train_graphs, validation_graphs, test_graphs = train_data['Gs'], validation_data['Gs'], test_data['Gs']
    all_graphs = train_graphs + validation_graphs + test_graphs
    all_seeds = train_seeds + validation_seeds + test_seeds
    preprocessor = PreProcessor(dim=params_man.embedding_dimensions, device=params_man.device, dtype=params_man.dtype)
    embedding_alg = get_embedding_algo(alg_name=params_man.embedding_alg_name, dim=params_man.embedding_dimensions)
    embeddings = get_embeddings_by_technique(params_man.technique, preprocessor, all_graphs, all_seeds, embedding_alg)

    return embeddings


def get_embeddings_by_technique(technique, preprocessor, all_graphs, all_seeds, embedding_alg):
    if technique in [Techniques.node_embedding_to_value, Techniques.node_embedding_to_routing,
                     Techniques.node_embedding_s_t_routing]:
        embeddings = preprocessor.compute_node_embeddings(all_graphs, all_seeds, embedding_alg)
    if technique == Techniques.graph_embedding_to_routing:
        embeddings = preprocessor.compute_graphs_embeddings(all_graphs, all_seeds, embedding_alg)

    return embeddings
    # if technique == Techniques.graph_embedding_to_rbc:
    #     embeddings = preprocessor.compute_graphs_embeddings(all_graphs, all_seeds, embedding_alg)
    #     all_rs = train_data['Rs'] + validation_data['Rs'] + test_data['Rs']
    #     all_ts = train_data['Ts'] + validation_data['Ts'] + test_data['Ts']
    #     eig_method, pi_max_err = params_man.learning_params[EmbStat.eigenvector_method], params_man.hyper_params[
    #         EmbStat.pi_max_err]
    #     device, dtype = params_man.device, params_man.dtype
    #     rbc_handler = RBC(eigenvector_method=eig_method, pi_max_error=pi_max_err, device=device, dtype=dtype)
    #     rbcs = [rbc_handler.compute_rbc(g, R, T) for g, R, T in zip(all_graphs, all_rs, all_ts)]
    #     embeddings = list(zip(embeddings, all_graphs, all_rs, all_ts, rbcs))


def split_embeddings(embeddings, train_len, validation_len, test_len):
    embeddings_train = embeddings[0: train_len]
    embeddings_validation = embeddings[train_len: train_len + validation_len]
    embeddings_test = embeddings[train_len + validation_len: train_len + validation_len + test_len]

    return embeddings_train, embeddings_validation, embeddings_test


def init_nn_model(param_embed):
    device = param_embed[EmbStat.device]
    dtype = param_embed[EmbStat.dtype]
    embed_dimension = param_embed[EmbStat.embd_dim]
    technique = param_embed['technique']

    if technique == Techniques.node_embedding_to_value:
        model = EmbeddingNeuralNetwork(embed_dimension, device, dtype)
    if technique == Techniques.node_embedding_s_t_routing:
        model = NeuralNetworkNodeEmbeddingSourceTargetRouting(embed_dimension, param_embed['num_nodes'], device, dtype)
    if technique == Techniques.node_embedding_to_routing:
        model = NeuralNetworkNodesEmbeddingRouting(embed_dimension, param_embed['num_nodes'], device, dtype)
    if technique == Techniques.graph_embedding_to_routing:
        model = NeuralNetworkGraphEmbeddingRouting(embed_dimension, param_embed['num_nodes'], device, dtype)
    if technique == Techniques.graph_embedding_to_rbc:
        model = NeuralNetworkGraphEmbeddingRbc(embed_dimension, param_embed['num_nodes'], device, dtype)

    return model


def train_model(nn_model, optimizer, p_man, train_data, embeddings_train, val_data, embeddings_validation):
    # init preprocessor
    device, dtype, dim = p_man.device, p_man.dtype, p_man.embedding_dimensions
    preprocessor = PreProcessor(dim=dim, device=device, dtype=dtype)

    # generate [feature, label] samples by technique (by default [[s,u,v,t], prob] samples)
    samples_train = generate_samples_by_technique(p_man, preprocessor, embeddings_train, train_data)
    samples_validation = generate_samples_by_technique(p_man, preprocessor, embeddings_validation, val_data)

    # training the model
    start_train_time = datetime.datetime.now()
    train_result = train_model_by_technique(nn_model, p_man, optimizer, samples_train, samples_validation)
    model_trained, train_error, validation_error = train_result
    total_train_time = datetime.datetime.now() - start_train_time
    print(f'train time: {total_train_time}')

    return model_trained, train_error, validation_error, total_train_time


def generate_samples_by_technique(p_man, preprocessor, embeddings, data):
    technique = p_man.technique
    if technique == Techniques.node_embedding_to_value:
        n_rand = p_man.n_rand_samples_graph
        samples = preprocessor.generate_random_samples(embeddings=embeddings, Rs=data['Rs'], num_rand_samples=n_rand)
    if technique == Techniques.node_embedding_s_t_routing:
        samples = preprocessor.generate_all_samples_s_t_routing(embeddings=embeddings, Rs=data['Rs'])
    if technique == Techniques.node_embedding_to_routing or technique == Techniques.graph_embedding_to_routing:
        samples = preprocessor.generate_all_samples_embeddings_to_routing(embeddings, data['Rs'])
    if technique == Techniques.graph_embedding_to_rbc:
        samples = preprocessor.generate_all_samples_embeddings_to_rbc(embeddings, data['Rs'])

    return samples


def train_model_by_technique(model, p_man, optimizer, samples_train, samples_val):
    technique = p_man.technique

    if technique == Techniques.node_embedding_to_value:
        train_res = EmbeddingML.train_model(model, samples_train, samples_val, p_man, optimizer)
    if technique == Techniques.node_embedding_s_t_routing:
        train_res = EmbeddingML.train_model_st_routing(model, samples_train, samples_val, p_man, optimizer)
    if technique == Techniques.node_embedding_to_routing or technique == Techniques.graph_embedding_to_routing:
        train_res = EmbeddingML.train_model_embed_to_routing(model, samples_train, samples_val, p_man, optimizer)
    if technique == Techniques.graph_embedding_to_rbc:
        train_res = EmbeddingML.train_model_embed_to_rbc(model, samples_train, samples_val, p_man, optimizer)

    return train_res


def test_model(model, p_man: EmbeddingsParams, test_data, test_embeddings):
    expected_rbcs, actual_rbcs = compute_rbcs(model, p_man, test_data, test_embeddings)
    euclidean_dist_median, kendall_tau_avg, spearman_avg, pearsonr_avg = calc_test_statistics(expected_rbcs,
                                                                                              actual_rbcs)
    np.set_printoptions(precision=3)
    # print(f'expected rbcs: {expected_rbcs}\n actual rbcs: {actual_rbcs}')

    return expected_rbcs, actual_rbcs, euclidean_dist_median, kendall_tau_avg, spearman_avg, pearsonr_avg


def compute_rbcs(model, p_man: EmbeddingsParams, test_data, test_embeddings):
    pi_err, eig_vec_mt = p_man.hyper_params[HyperParams.pi_max_err], p_man.learning_params[EmbStat.eigenvector_method]
    device, dtype = p_man.device, p_man.dtype
    rbc_handler = RBC(eigenvector_method=eig_vec_mt, pi_max_error=pi_err, device=device, dtype=dtype)

    expected_rbcs = []
    actual_rbcs = []
    print(f'predicting routing policy and compute rbcs')
    num_gs = len(test_data['Gs'])
    i = 0
    for R, T, g, test_embedding in zip(test_data['Rs'], test_data['Ts'], test_data['Gs'], test_embeddings):
        i += 1
        print(f'{i} out of {num_gs}')
        predicted_r_policy = predict_routing_policy(p_man.technique, model, test_embedding, p_man)
        expected_rbc = rbc_handler.compute_rbc(g, R, T).cpu().detach().numpy()
        expected_rbcs.append(expected_rbc)
        actual_rbc = rbc_handler.compute_rbc(g, predicted_r_policy, T).cpu().detach().numpy()
        actual_rbcs.append(actual_rbc)

    return expected_rbcs, actual_rbcs


def predict_routing_policy(technique, model, test_embedding, p_man):
    if technique == Techniques.node_embedding_to_value:
        test_r_policy = EmbeddingML.predict_routing(model, test_embedding, p_man)
    if technique == Techniques.node_embedding_s_t_routing:
        test_r_policy = EmbeddingML.predict_s_t_routing(model, test_embedding, p_man)
    if technique == Techniques.node_embedding_to_routing:
        test_r_policy = EmbeddingML.predict_routing_all_nodes_embed(model, test_embedding, p_man)
    if technique == Techniques.graph_embedding_to_routing or technique == Techniques.graph_embedding_to_rbc:
        test_r_policy = EmbeddingML.predict_graph_embedding(model, test_embedding, p_man)

    return test_r_policy


def calc_test_statistics(expected_rbcs, actual_rbcs):
    rbcs_zip = list(zip(expected_rbcs, actual_rbcs))

    euclidean_arr = np.array([scipy.spatial.distance.euclidean(expected, actual) for expected, actual in rbcs_zip])
    kendall_arr = np.array([scipy.stats.kendalltau(expected, actual)[0] for expected, actual in rbcs_zip])
    spearman_arr = np.array([scipy.stats.spearmanr(expected, actual)[0] for expected, actual in rbcs_zip])
    pearsonr_arr = np.array([scipy.stats.pearsonr(expected, actual)[0] for expected, actual in rbcs_zip])

    euclidean_median = np.median(euclidean_arr)
    kendall_avg, spearman_avg, pearsonr_avg = kendall_arr.mean(), spearman_arr.mean(), pearsonr_arr.mean()
    print(f'euclidean dist median: {euclidean_median} \nkendall avg:{kendall_avg} \nspearman avg: {spearman_avg} \n'
          f'pearson avg: {pearsonr_avg}')

    return euclidean_median, kendall_avg, spearman_avg, pearsonr_avg


def update_params_man(params_man, train_val_test, train_res, test_res, optimizer):
    train_data, validation_data, test_data = train_val_test
    trained_model, train_err, validation_err, train_time,  = train_res
    expected_rbcs, actual_rbcs, euclidean_dist_median, kendall_tau_avg, spearman_avg, pearsonr_avg = test_res

    params_man.n_graphs_train = int(len(train_data['Gs']) / params_man.seeds_per_train_graph)
    params_man.n_graphs_validation = len(validation_data['Gs'])
    params_man.n_graphs_test = len(test_data['Gs'])
    params_man.trained_model = trained_model
    params_man.train_runtime = train_time
    params_man.train_error = train_err
    params_man.validation_error = validation_err
    params_man.expected_rbcs = str(expected_rbcs)
    params_man.actual_rbcs = str(actual_rbcs)
    params_man.euclidean_dis_median = euclidean_dist_median
    params_man.kendall_tau_avg = kendall_tau_avg
    params_man.spearman_avg = spearman_avg
    params_man.pearson_avg = pearsonr_avg
    params_man.network_structure = trained_model.linear_relu_stack.__str__()
    params_man.optimizer_params = optimizer.get_optimizer_params()


if __name__ == '__main__':
    random.seed(42)
    csv_save_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Statistics\statistics.csv'
    trained_models_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\TrainedModels'

    num_nodes = 11
    n_seeds_train_graph = 1
    path_obj = Paths.Single_Graph_Fixed_Routing_SPBC_11_nodes()

    params_statistics1 = {
        EmbStat.centrality: Centralities.SPBC,
        EmbStat.embd_dim: num_nodes - 1,
        EmbStat.embedding_alg: EmbeddingAlgorithms.glee,
        'seed_range': 10000,
        'technique': Techniques.node_embedding_to_value,
        HyperParams.optimizer: OptimizerTypes.Adam,
        HyperParams.learning_rate: 1e-4,
        HyperParams.epochs: 80,
        HyperParams.batch_size: 2048,
        HyperParams.weight_decay: 0.0001,
        HyperParams.momentum: 0.0,
        HyperParams.error_type: ErrorTypes.mse,
        EmbStat.n_random_samples_per_graph: NumRandomSamples.N_power_2,
        'path_obj': path_obj,
        'num_nodes': num_nodes,
        EmbStat.device: TorchDevice.gpu,
        EmbStat.dtype: TorchDtype.float,
        EmbStat.csv_save_path: csv_save_path,
        EmbeddingOutputs.graphs_root_path: path_obj.root_path,
        EmbeddingOutputs.trained_model_root_path: trained_models_path,
        EmbStat.n_seeds_train_graph: n_seeds_train_graph,
        EmbStat.eigenvector_method: EigenvectorMethod.torch_eig,
        HyperParams.pi_max_err: 0.00001
    }


    n_seeds_lst = [1, 10, 30, 70, 100]
    for train_seeds in n_seeds_lst:
        params_statistics1[EmbStat.n_seeds_train_graph] = train_seeds
        run_test(pr_st=params_statistics1)




