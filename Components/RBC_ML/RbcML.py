import datetime

import networkx as nx
import torch
from Utils.CommonStr import HyperParams
from Components.RBC_ML.RbcNetwork import RbcNetwork
from Utils.CommonStr import LearningParams
from Utils.CommonStr import ErrorTypes
from Utils.CommonStr import OptimizerTypes


def learn_models(learning_params, nodes_mapping_reverse):
    zero_mat, const_mat, traffic_paths = get_fixed_mat(learning_params, nodes_mapping_reverse)
    num_nodes = len(learning_params[LearningParams.adjacency_matrix][0])
    use_sigmoid = learning_params[LearningParams.sigmoid]
    pi_max_err = learning_params[LearningParams.hyper_parameters][HyperParams.pi_max_err]
    eigenvector_method = learning_params[LearningParams.eigenvector_method]
    zero_traffic = learning_params[LearningParams.zeros_no_path_traffic_matrix]
    device = learning_params[LearningParams.device]
    dtype = learning_params[LearningParams.dtype]
    model = RbcNetwork(num_nodes=num_nodes, use_sigmoid=use_sigmoid, pi_max_err=pi_max_err,
                       eigenvector_method=eigenvector_method, device=device, dtype=dtype, zero_traffic=zero_traffic)

    hyper_params = learning_params[LearningParams.hyper_parameters]
    criterion = get_criterion(hyper_params[HyperParams.error_type])
    optimizer = get_optimizer(model, hyper_params[HyperParams.optimizer], hyper_params[HyperParams.learning_rate],
                              hyper_params[HyperParams.momentum])

    start_time = datetime.datetime.now()

    for t in range(hyper_params[HyperParams.epochs]):
        y_pred = model(zero_mat, const_mat, traffic_paths)
        loss = criterion(y_pred, learning_params[LearningParams.target])
        print(t, loss.item()) if t % 1 == 0 else None
        optimizer.zero_grad()
        loss.backward()   # backward unable handle 50 nodes
        optimizer.step()

    print(f'\nRun Time -  {datetime.datetime.now() - start_time} '
          f'\n\nLearning Target - {learning_params[LearningParams.target]}')
    model_t = model.weights_t
    model_r = (torch.sigmoid(model.weights_r) * zero_mat + const_mat) if learning_params[LearningParams.sigmoid] else \
              (model.weights_r * zero_mat) + const_mat

    return model_t, model_r, loss.item()


def get_fixed_mat(learning_params, nodes_mapping_reverse):

    adj_matrix_t = learning_params[LearningParams.adjacency_matrix].t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
    adj_size = learning_params[LearningParams.adjacency_matrix].size()[0]
    device = learning_params[LearningParams.device]
    dtype = learning_params[LearningParams.dtype]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=dtype, device=device)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=dtype, device=device)
    path_exist = torch.full(size=(adj_size, adj_size), fill_value=0, dtype=dtype, device=device)

    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1 if learning_params[LearningParams.src_src_one] else 0
            zeros_mat[s, t] = adj_matrix_t  # putting zeros where the transposed adj_matrix has zeros
            zeros_mat[s, t, s] = 0 if learning_params[LearningParams.src_row_zeros] else zeros_mat[s, t, s]
            zeros_mat[s, t, :, t] = 0 if learning_params[LearningParams.target_col_zeros] else zeros_mat[s, t, :, t]

            if s == t:
                try:
                    nx.find_cycle(g, nodes_mapping_reverse[s])
                    path_exist[s, t] = 1.0
                except:
                    path_exist[s, t] = 0.0
            else:
                if nx.has_path(g, nodes_mapping_reverse[s], nodes_mapping_reverse[t]):
                    path_exist[s, t] = 1.0
                else:
                    path_exist[s, t] = 0.0

    return zeros_mat, constant_mat, path_exist


def get_criterion(error_type):
    if error_type == ErrorTypes.mse:
        return torch.nn.MSELoss(reduction='sum')
    else:
        return torch.nn.MSELoss(reduction='sum')


def get_optimizer(model, optimizer_type, learning_rate, momentum):
    if optimizer_type == OptimizerTypes.AdaDelta:
        return torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.AdaGrad:
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.Adam:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.AdamW:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.SparseAdam:
        return torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.AdaMax:
        return torch.optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.ASGD:
        return torch.optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.LBFGS:
        return torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.RmsProp:
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, eps=1e-05, centered=True)
    elif optimizer_type == OptimizerTypes.Rprop:
        return torch.optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.SGD:
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


