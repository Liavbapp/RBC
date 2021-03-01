import datetime
import networkx as nx
import torch
from Components.RBC_ML.Optimizer import Optimizer
from Utils.CommonStr import HyperParams
from Components.RBC_ML.RbcNetwork import RbcNetwork
from Utils.CommonStr import LearningParams
from Utils.CommonStr import ErrorTypes


def learn_models(model, g, learning_params, nodes_mapping_reverse, optimizer):

    hyper_params = learning_params[LearningParams.hyper_parameters]
    loss_criterion = get_criterion(hyper_params[HyperParams.error_type])
    consider_traffic_paths = learning_params[LearningParams.consider_traffic_paths]
    zero_mat, const_mat, traffic_paths = get_fixed_mat(g, learning_params, nodes_mapping_reverse)

    start_time = datetime.datetime.now()
    changed =False
    for t in range(hyper_params[HyperParams.epochs]):
        y_pred = model(zero_mat, const_mat, traffic_paths)
        loss = loss_criterion(y_pred, learning_params[LearningParams.target])
        print(t, loss.item()) if t % 1 == 0 else None
        optimizer.zero_grad()
        loss.backward()  # backward unable handle 50 nodes
        optimizer.step()
        # if loss.item() < 0.05 and not changed:
        #     optimizer.change_learning_rate(1e-6)
        #     changed = True

    print(f'\nRun Time -  {datetime.datetime.now() - start_time} '
          f'\n\nLearning Target - {learning_params[LearningParams.target]}')
    model_t = torch.mul(model.weights_t, traffic_paths) if consider_traffic_paths else model.weights_t
    model_r = (torch.sigmoid(model.weights_r) * zero_mat + const_mat) if learning_params[LearningParams.sigmoid] else \
        (model.weights_r * zero_mat) + const_mat

    return model_t, model_r, loss.item()


def get_fixed_mat(g, learning_params, nodes_mapping_reverse):
    adj_matrix_t = learning_params[
        LearningParams.adjacency_matrix].t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
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



