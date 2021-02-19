class RbcMatrices:
    adjacency_matrix = 'Adjacency Matrix'
    routing_policy = "Routing Policy"
    traffic_matrix = "Traffic Matrix"


class HyperParams:
    epochs = "Epochs"
    learning_rate = "Learning Rate"
    momentum = "Momentum"
    optimizer = "Optimizer"
    pi_max_err = "PI Max Error"
    error_type = "Error Type"


class StatisticsParams:
    centrality = "Centrality Type"
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
    cols = [centrality, num_nodes, num_edges, target, prediction,
            error, error_type, sigmoid, src_src_one, src_row_zeros, target_col_zeros, runtime, learning_rate,
            epochs, momentum, optimizer, path]


class LearningParams:
    hyper_parameters = "Hyper Parameters"
    adjacency_matrix = RbcMatrices.adjacency_matrix
    target = StatisticsParams.target
    src_src_one = StatisticsParams.src_src_one
    src_row_zeros = StatisticsParams.src_row_zeros
    target_col_zeros = StatisticsParams.target_col_zeros
    sigmoid = StatisticsParams.sigmoid


class OptimizerTypes:
    adam = "Adam"
    sgd = "SGD"
    asgd = "ASGD"


class ErrorTypes:
    mse = "MSE"
