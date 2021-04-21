import datetime
import os
import random
import sys
import torch
import numpy as np
import networkx as nx
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
    trained_model, train_time, train_err = train_res

    # testing
    test_res = test_model(trained_model, params_man, test_data, embed_test)
    expected_rbc, actual_rbc, test_routing_policy, rbc_diff = test_res

    # updating the test running info
    update_params_man(params_man, split_data_res, train_res, test_res, optimizer)


def init_optimizer(pr_st, nn_model):
    optimizer_name, lr, momentum = pr_st[EmbStat.optimizer], pr_st[EmbStat.learning_rate], pr_st[HyperParams.momentum]
    wd = pr_st[HyperParams.weight_decay]
    optimizer = Optimizer(model=nn_model, name=optimizer_name, learning_rate=lr, momentum=momentum, weight_decay=wd)

    return optimizer


def split_to_train_validation_test(p_man):
    Rs, Ts, Gs, path_params = extract_info_from_path(p_man.graph_paths, p_man)
    train_data_lists = {'Rs': Rs[3:], 'Ts': Ts[3:], 'Gs': Gs[3:], 'path_params': path_params[3:]}
    validation_data_lists = {'Rs': Rs[1:3], 'Ts': Ts[1:3], 'Gs': Gs[1:3], 'path_params': path_params[1:3]}
    test_data_lists = {'Rs': [Rs[0]], 'Ts': [Ts[0]], 'Gs': [Gs[0]], 'path_params': [path_params[0]]}
    # validation_data_lists = test_data_lists  # TODO: delete this, only for debugging

    return train_data_lists, validation_data_lists, test_data_lists


def extract_info_from_path(paths, p_man):
    routing_np, traffic_np, adj_np = 'routing_policy.npy', 'traffic_mat.npy', 'adj_mat.npy'
    R_lst = [torch.tensor(np.load(f'{path}\\{routing_np}'), dtype=p_man.dtype, device=p_man.device) for path in paths]
    T_lst = [torch.tensor(np.load(f'{path}\\{traffic_np}'), dtype=p_man.dtype, device=p_man.device) for path in paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(f'{path}\\{adj_np}')) for path in paths]
    seeds = random.sample(range(0, p_man.seed_range), len(paths))
    path_params = [{path: {EmbeddingPathParams.seed: seed}} for path, seed in zip(paths, seeds)]

    return R_lst, T_lst, G_lst, path_params


def generate_embeddings(train_data, validation_data, test_data, params_man):
    train_rs_len, val_rs_len, test_rs_len = len(train_data['Rs']), len(validation_data['Rs']), len(test_data['Rs'])
    combined_embed = get_combined_embeddings(train_data, validation_data, test_data, params_man)
    embed_train, embed_validation, embed_test = split_embeddings(combined_embed, train_rs_len, val_rs_len, test_rs_len)

    return embed_train, embed_validation, embed_test


def get_combined_embeddings(train_data, validation_data, test_data, params_man):
    technique = params_man.technique
    train_seeds, validation_seeds, test_seeds = generate_seeds(train_data, validation_data, test_data)
    train_graphs, validation_graphs, test_graphs = train_data['Gs'], validation_data['Gs'], test_data['Gs']
    all_graphs = train_graphs + validation_graphs + test_graphs
    all_seeds = train_seeds + validation_seeds + test_seeds
    preprocessor = PreProcessor(dim=params_man.embedding_dimensions, device=params_man.device, dtype=params_man.dtype)
    embedding_alg = get_embedding_algo(alg_name=params_man.embedding_alg_name, dim=params_man.embedding_dimensions)

    if technique in [Techniques.node_embedding_to_value, Techniques.node_embedding_to_routing,
                     Techniques.node_embedding_s_t_routing]:
        embeddings = preprocessor.compute_node_embeddings(all_graphs, all_seeds, embedding_alg)
    if technique == Techniques.graph_embedding_to_routing:
        embeddings = preprocessor.compute_graphs_embeddings(all_graphs, all_seeds, embedding_alg)
    if technique == Techniques.graph_embedding_to_rbc:
        embeddings = preprocessor.compute_graphs_embeddings(all_graphs, all_seeds, embedding_alg)
        all_rs = train_data['Rs'] + validation_data['Rs'] + test_data['Rs']
        all_ts = train_data['Ts'] + validation_data['Ts'] + test_data['Ts']
        eig_method, pi_max_err = params_man.learning_params[EmbStat.eigenvector_method], params_man.hyper_params[
            EmbStat.pi_max_err]
        device, dtype = params_man.device, params_man.dtype
        rbc_handler = RBC(eigenvector_method=eig_method, pi_max_error=pi_max_err, device=device, dtype=dtype)
        rbcs = [rbc_handler.compute_rbc(g, R, T) for g, R, T in zip(all_graphs, all_rs, all_ts)]
        embeddings = list(zip(embeddings, all_graphs, all_rs, all_ts, rbcs))

    return embeddings


def generate_seeds(train_data, validation_data, test_data):
    seed_str = EmbeddingPathParams.seed
    train_seeds = [list(param_info.values())[0][seed_str] for param_info in train_data['path_params']]
    validation_seeds = [list(param_info.values())[0][seed_str] for param_info in validation_data['path_params']]
    test_seeds = [list(param_info.values())[0][seed_str] for param_info in test_data['path_params']]

    return train_seeds, validation_seeds, test_seeds


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
    device = p_man.device
    dtype = p_man.dtype
    dim = p_man.embedding_dimensions
    technique = p_man.technique
    preprocessor = PreProcessor(dim=dim, device=device, dtype=dtype)

    if technique == Techniques.node_embedding_to_value:
        samples_train = preprocessor.generate_random_samples(embeddings=embeddings_train, Rs=train_data['Rs'], num_rand_samples=p_man.num_random_samples_graph)
        # samples_train = random.sample(samples_train, int(len(samples_train) * 0.1))
        samples_validation = preprocessor.generate_all_samples(embeddings=embeddings_validation, Rs=val_data['Rs'])
        start_time = datetime.datetime.now()
        model_trained, train_error = EmbeddingML.train_model(nn_model, samples_train, samples_validation, p_man,
                                                             optimizer)
    #
    # if technique == Techniques.node_embedding_s_t_routing:
    #     samples_train = preprocessor.generate_all_samples_s_t_routing(embeddings=embeddings_train, Rs=train_data['Rs'])
    #     samples_validation = preprocessor.generate_all_samples_s_t_routing(embeddings=embeddings_validation, Rs=val_data['Rs'])
    #     start_time = datetime.datetime.now()
    #     model_trained, train_error = EmbeddingML.train_model_s_t_routing(nn_model, samples_train, samples_validation, p_man, optimizer)
    #
    # if technique == Techniques.node_embedding_to_routing or technique == Techniques.graph_embedding_to_routing:
    #     samples_train = preprocessor.generate_all_samples_embeddings_to_routing(embeddings_train, train_data['Rs'])
    #     samples_validation = preprocessor.generate_all_samples_embeddings_to_routing(embeddings_validation, val_data['Rs'])
    #     start_time = datetime.datetime.now()
    #     model_trained, train_error = EmbeddingML.train_model_embed_to_routing(nn_model, samples_train, samples_validation, p_man, optimizer)
    #
    # if technique == Techniques.graph_embedding_to_rbc:
    #     samples_train = preprocessor.generate_all_samples_embeddings_to_rbc(embeddings_train, train_data['Rs'])
    #     samples_validation = preprocessor.generate_all_samples_embeddings_to_rbc(embeddings_validation, val_data['Rs'])
    #     start_time = datetime.datetime.now()
    #     model_trained, train_error = EmbeddingML.train_model_embed_to_rbc(nn_model, samples_train, samples_validation, p_man, optimizer)

    train_time = datetime.datetime.now() - start_time
    print(f'train time: {train_time}')

    return model_trained, train_time, train_error


def test_model(model, p_man: EmbeddingsParams, test_data, test_embeddings):
    pi_max_err = p_man.hyper_params[HyperParams.pi_max_err]
    eigine_vec_method = p_man.learning_params[EmbStat.eigenvector_method]
    device, dtype = p_man.device, p_man.dtype
    technique = p_man.technique

    model.eval()  # set the model for testing
    test_example = (test_data['Rs'][0], test_data['Ts'][0], test_data['Gs'][0])  # todo: generalize it..
    test_embedding = test_embeddings[0]  # todo: generalize it..
    test_R, test_T, test_G = test_example
    # train_R, train_T, train_G = train_example

    rbc_train = RBC(eigenvector_method=eigine_vec_method, pi_max_error=pi_max_err, device=device, dtype=dtype)
    rbc_test = RBC(eigenvector_method=eigine_vec_method, pi_max_error=pi_max_err, device=device, dtype=dtype)

    expected_rbc = rbc_train.compute_rbc(test_G, test_R, test_T)
    if technique == Techniques.node_embedding_to_value:
        test_r_policy = EmbeddingML.predict_routing(model, test_embedding, p_man)
    if technique == Techniques.node_embedding_s_t_routing:
        test_r_policy = EmbeddingML.predict_s_t_routing(model, test_embedding, p_man)
    if technique == Techniques.node_embedding_to_routing:
        test_r_policy = EmbeddingML.predict_routing_all_nodes_embed(model, test_embedding, p_man)
    if technique == Techniques.graph_embedding_to_routing or technique == Techniques.graph_embedding_to_rbc:
        test_r_policy = EmbeddingML.predict_graph_embedding(model, test_embedding, p_man)

    actual_rbc = rbc_test.compute_rbc(test_G, test_r_policy, test_T)
    rbc_diff = calc_rbc_diff(expected_rbc, actual_rbc)
    print(f'expected rbc: {expected_rbc}')
    print(f'actual rbc: {actual_rbc}')
    print(f'rbc diff: {rbc_diff}')

    return expected_rbc, actual_rbc, test_r_policy, rbc_diff


def update_params_man(params_man, train_val_test, train_res, test_res, optimizer):
    train_data, _, test_data = train_val_test
    trained_model, train_time, train_err = train_res
    expected_rbc, actual_rbc, test_routing_policy, rbc_diff = test_res

    params_man.train_path_params = train_data['path_params']
    params_man.test_path_params = test_data['path_params']
    params_man.trained_model = trained_model
    params_man.train_runtime = train_time
    params_man.train_error = train_err
    params_man.expected_rbc = expected_rbc.data
    params_man.actual_rbc = actual_rbc.data
    params_man.test_routing_policy = test_routing_policy
    params_man.test_graph = test_data['Gs'][0]  # TODO: genralize it to many test data...
    params_man.rbc_diff = rbc_diff
    params_man.network_structure = trained_model.linear_relu_stack.__str__()
    params_man.optimizer_params = optimizer.get_optimizer_params()
    params_man.prepare_params_statistics()
    params_man.save_params_statistics()


def calc_rbc_diff(expected_rbc, actual_rbc):
    return ((expected_rbc - actual_rbc) ** 2).sum(axis=0).item()


if __name__ == '__main__':
    random.seed(42)
    csv_save_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Results\LearningEmbedding.csv'
    embedding_outputs_root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\9_nodes_fixed_rbc\Embeddings_Results'

    num_nodes = 9
    n_seeds_routing = 10
    graph_desc = Paths.Single_Graph_Fixed_Routing_SPBC_9_nodes(n_seeds_routing)

    params_statistics1 = {
        EmbStat.centrality: Centralities.SPBC,
        EmbStat.device: TorchDevice.gpu,
        EmbStat.dtype: TorchDtype.float,
        EmbStat.embd_dim: num_nodes - 1,
        EmbStat.embedding_alg: EmbeddingAlgorithms.laplacian_eigenmaps,
        EmbStat.n_graphs: graph_desc.n_graphs,
        EmbStat.n_seeds_graph: graph_desc.total_seeds_per_graph,
        EmbStat.n_routing_graph: graph_desc.n_routing_per_graph,
        EmbStat.routing_type: graph_desc.routing_type,
        EmbStat.graphs_desc: graph_desc,
        EmbStat.n_random_samples_graph: NumRandomSamples.N_power_2,
        'seed_range': 3000,
        'num_nodes': num_nodes,
        'technique': Techniques.node_embedding_to_value,
        EmbStat.csv_save_path: csv_save_path,
        EmbeddingOutputs.root_path: embedding_outputs_root_path,
        HyperParams.optimizer: OptimizerTypes.Adam,
        HyperParams.learning_rate: 1e-4,
        HyperParams.epochs: 120,
        HyperParams.batch_size: 128,
        HyperParams.weight_decay: 0.00000,
        HyperParams.momentum: 0.0,
        HyperParams.pi_max_err: 0.00001,
        HyperParams.error_type: ErrorTypes.mse,
        EmbStat.eigenvector_method: EigenvectorMethod.torch_eig
    }

    num_nodes_arr = [9, 11]
    embd_algs_lst = [EmbeddingAlgorithms.glee, EmbeddingAlgorithms.laplacian_eigenmaps, EmbeddingAlgorithms.socio_dim, EmbeddingAlgorithms.node2vec]
    n_seeds_routing_lst = [10, 20, 30, 40, 50, 70, 90, 100]

    combinations = list(itertools.product(num_nodes_arr, embd_algs_lst, n_seeds_routing_lst))
    num_combinations = len(combinations)
    i = 0
    for num_nodes_i, embd_alg, n_seed_routing_i in combinations:
        i += 1
        print(f'{i} out of {num_combinations}')

        num_nodes = num_nodes_i
        if num_nodes_i == 11:
            embedding_outputs_root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\11_nodes_fixed_rbc\Embeddings_Results'
            graph_desc = Paths.Single_Graph_Fixed_Routing_SPBC_11_nodes(n_seeds_routing)
        if num_nodes_i == 9:
            embedding_outputs_root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\9_nodes_fixed_rbc\Embeddings_Results'
            graph_desc = Paths.Single_Graph_Fixed_Routing_SPBC_9_nodes(n_seeds_routing)



        params_statistics1[EmbStat.embd_dim] = num_nodes - 1
        params_statistics1[EmbStat.embedding_alg] = embd_alg
        params_statistics1[EmbStat.n_graphs] = graph_desc.n_graphs
        params_statistics1[EmbStat.n_seeds_graph] = graph_desc.total_seeds_per_graph
        params_statistics1[EmbStat.n_routing_graph] = graph_desc.n_routing_per_graph
        params_statistics1[EmbStat.routing_type] = graph_desc.routing_type
        params_statistics1[EmbStat.graphs_desc] = graph_desc
        run_test(pr_st=params_statistics1)
