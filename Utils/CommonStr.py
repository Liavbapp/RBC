import torch


class RbcMatrices:
    name = "RBC Matrices"
    root_path = "Root Path"
    adjacency_matrix = 'Adjacency Matrix'
    routing_policy = "Routing Policy"
    traffic_matrix = "Traffic Matrix"


class EmbeddingOutputs:
    test_routing_policy = 'Test Routing Policy'
    root_path = "Root Path"
    test_graph = 'Test graph'
    trained_model = "Model"
    train_path_params = "Train path params"
    test_path_params = "Test path params"


class HyperParams:
    name = "Hyper Params"
    epochs = "Epochs"
    learning_rate = "Learning Rate"
    momentum = "Momentum"
    weight_decay = "Weight Decay"
    optimizer = "Optimizer"
    pi_max_err = "PI Max Error"
    error_type = "Error Type"
    batch_size = "Batch Size"


class EmbeddingStatistics:
    name = "EmbeddingsStatistics"
    id = "ID"
    centrality = "Centrality Type"
    centrality_params = "Centrality Params"
    embd_dim = "Embedding Dimensions"
    rbc_target = "RBC Target"
    rbc_diff = "RBC Diff"
    rbc_test = "RBC Test"
    train_error = "Train Error"
    error_type = "Error Type"
    network_structure = "Network Structure"
    train_runtime = "Train RunTime"
    embedding_alg = "Embedding Alg"
    learning_rate = HyperParams.learning_rate
    epochs = HyperParams.epochs
    weight_decay = HyperParams.weight_decay
    optimizer = HyperParams.optimizer
    batch_size = HyperParams.batch_size
    optimizer_params = "optimizer params"
    eigenvector_method = "Eigenvector Computing Method "
    pi_max_err = HyperParams.pi_max_err
    path = "Path"
    comments = "Comments"
    device = "Torch Device"
    dtype = "Torch Dtype"
    csv_save_path = "Saving Csv Path"
    graphs_desc = "Graph desc"

    cols = [id, centrality, centrality_params, graphs_desc, embedding_alg, embd_dim, rbc_target, rbc_test,
            rbc_diff, train_error, error_type, network_structure, train_runtime, learning_rate, epochs, batch_size, weight_decay,
            optimizer, optimizer_params, eigenvector_method, pi_max_err, path, comments, device, dtype]


class StatisticsParams:
    name = "Statistics Params"
    id = "ID"
    centrality = "Centrality Type"
    centrality_params = "Centrality Params"
    num_nodes = "Num Nodes"
    num_edges = "Num Edges"
    target = "Target"
    prediction = "Prediction"
    error = "Error"
    error_type = "Error Type"
    sigmoid = "With Sigmoid"
    src_src_one = "Predecessor[src, src]=1"
    src_row_zeros = "Predecessor[src, :]=0"
    target_col_zeros = "Predecessor[:, target]=0"
    fixed_R = "Fixed R policy"
    fixed_T = "Fixed T Matrix"
    runtime = "RunTime"
    learning_rate = HyperParams.learning_rate
    epochs = HyperParams.epochs
    weight_decay = HyperParams.weight_decay
    momentum = HyperParams.momentum
    optimizer = HyperParams.optimizer
    pi_max_err = HyperParams.pi_max_err

    path = "Path"
    comments = "Comments"
    eigenvector_method = "Eigenvector Computing Method "
    device = "Torch Device"
    dtype = "Torch Dtype"
    consider_traffic_paths = "Traffic Paths"
    optimizer_params = "optimizer params"
    csv_save_path = "Saving csv Path"

    cols = [id, centrality, centrality_params, num_nodes, num_edges, epochs, learning_rate, weight_decay,
            optimizer, optimizer_params, eigenvector_method, pi_max_err, sigmoid, src_src_one, src_row_zeros,
            target_col_zeros, fixed_T, fixed_R, consider_traffic_paths, device, dtype, path, comments, target
        , prediction, error, error_type, runtime]


class LearningParams:
    name = "Learning Params"
    hyper_parameters = "Hyper Parameters"
    adjacency_matrix = RbcMatrices.adjacency_matrix
    target = StatisticsParams.target
    src_src_one = StatisticsParams.src_src_one
    src_row_zeros = StatisticsParams.src_row_zeros
    target_col_zeros = StatisticsParams.target_col_zeros
    sigmoid = StatisticsParams.sigmoid
    fixed_R = StatisticsParams.fixed_R
    fixed_T = StatisticsParams.fixed_T
    eigenvector_method = StatisticsParams.eigenvector_method
    device = StatisticsParams.device
    dtype = StatisticsParams.dtype
    consider_traffic_paths = StatisticsParams.consider_traffic_paths
    centrality_params = StatisticsParams.centrality_params


class TorchDevice:
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:0')


class TorchDtype:
    float = torch.float


class EigenvectorMethod:
    power_iteration = "Power Iteration"
    torch_eig = "Torch eig"


class OptimizerTypes:
    Rprop = "Rprop"
    RmsProp = "RmsProp"
    LBFGS = "LBFGS"
    AdaMax = "AdaMax"
    SparseAdam = "SparseAdam"
    AdaGrad = "AdaGrad"
    AdaDelta = "AdaDelta"
    Adam = "Adam"
    SGD = "SGD"
    ASGD = "ASGD"
    AdamW = "ADAMW"


class Centralities:
    name = "Centralities"
    SPBC = "SPBC"
    Degree = "Degree"
    Closeness = "Closeness"
    Eigenvector = "Eigenvector"
    Load = "Load"


class ErrorTypes:
    mse = "MSE"


class EmbeddingPathParams:
    seed = 'seed'


class EmbeddingAlgorithms:
    node2vec = "Node2Vec"
    diff2vec = "Diff2Vec"
    rand_ne = "RandNE"
    glee = "GLEE"
    net_mf = "NetMF"
    nnsed = "NNSED"
    danmf = "DANMF"
    mnmf = "MNMF"
    big_clam = "BigClam"
    symm_nmf = "SymmNMF"
    socio_dim = "SocioDim"
    node_sketch = "NodeSketch"
    boost_ne = "BoostNE"
    walklets = "Walklets"
    gra_rep = "GraRep"
    nmfadmm = "NMFADMM"
    laplacian_eigenmaps = "LaplacianEigenmaps"
    feather_node = "FeatherNode"
    ae = "AE"
    deep_walk = "DeepWalk"
    graph_wave = "GraphWave"
    musae = "MUSAE"
    role2vec = "Role2Vec"
    gl2vec = "GL2Vec"
