import torch


class RbcMatrices:
    name = "RBC Matrices"
    adjacency_matrix = 'Adjacency Matrix'
    routing_policy = "Routing Policy"
    traffic_matrix = "Traffic Matrix"


class HyperParams:
    name = "Hyper Params"
    epochs = "Epochs"
    learning_rate = "Learning Rate"
    momentum = "Momentum"
    optimizer = "Optimizer"
    pi_max_err = "PI Max Error"
    error_type = "Error Type"


class StatisticsParams:
    name = "Statistics Params"
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
    runtime = "RunTime"
    learning_rate = HyperParams.learning_rate
    epochs = HyperParams.epochs
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

    cols = [centrality, num_nodes, num_edges, target, prediction,
            error, error_type, sigmoid, src_src_one, src_row_zeros, target_col_zeros, runtime, learning_rate,
            epochs, optimizer, optimizer_params, path, comments, eigenvector_method, device, dtype, consider_traffic_paths]


class LearningParams:
    name = "Learning Params"
    hyper_parameters = "Hyper Parameters"
    adjacency_matrix = RbcMatrices.adjacency_matrix
    target = StatisticsParams.target
    src_src_one = StatisticsParams.src_src_one
    src_row_zeros = StatisticsParams.src_row_zeros
    target_col_zeros = StatisticsParams.target_col_zeros
    sigmoid = StatisticsParams.sigmoid
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


class ErrorTypes:

    mse = "MSE"
