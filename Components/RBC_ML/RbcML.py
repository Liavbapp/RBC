import datetime
import torch
from Utils.CommonStr import HyperParams
from Components.RBC_ML.RbcNetwork import RbcNetwork
from Utils.CommonStr import LearningParams
from Utils.CommonStr import ErrorTypes
from Utils.CommonStr import OptimizerTypes
DTYPE = torch.float
DEVICE = torch.device("cuda:0")


def learn_models(learning_params):
    zero_mat, const_mat = get_fixed_mat(learning_params)
    model = RbcNetwork(learning_params[LearningParams.adjacency_matrix],  learning_params[LearningParams.sigmoid])

    hyper_params = learning_params[LearningParams.hyper_parameters]
    criterion = get_criterion(hyper_params[HyperParams.error_type])
    optimizer = get_optimizer(model, hyper_params[HyperParams.optimizer], hyper_params[HyperParams.learning_rate],
                              hyper_params[HyperParams.momentum])

    start_time = datetime.datetime.now()

    for t in range(hyper_params[HyperParams.epochs]):
        y_pred = model(learning_params[LearningParams.adjacency_matrix], zero_mat, const_mat, hyper_params[HyperParams.pi_max_err])
        loss = criterion(y_pred, learning_params[LearningParams.target])
        print(t, loss.item()) if t % 100 == 0 else None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\nRun Time -  {datetime.datetime.now() - start_time} '
          f'\n\nLearning Target - {learning_params[LearningParams.target]}')
    model_t = model.weights_t
    model_r = torch.sigmoid(model.weights_r * zero_mat + const_mat)
    return model_t, model_r, loss.item()


def get_fixed_mat(learning_params):

    adj_matrix_t = learning_params[LearningParams.adjacency_matrix].t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
    adj_size = learning_params[LearningParams.adjacency_matrix].size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)

    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1 if learning_params[LearningParams.src_src_one] else 0
            zeros_mat[s, t] = adj_matrix_t  # putting zeros where the transposed adj_matrix has zeros
            zeros_mat[s, t, s] = 0 if learning_params[LearningParams.src_row_zeros] else zeros_mat[s, t, s]
            zeros_mat[s, t, :, t] = 0 if learning_params[LearningParams.target_col_zeros] else zeros_mat[s, t, :, t]

    return zeros_mat, constant_mat


def get_criterion(error_type):
    if error_type == ErrorTypes.mse:
        return torch.nn.MSELoss(reduction='sum')
    else:
        return torch.nn.MSELoss(reduction='sum')


def get_optimizer(model, optimizer_type, learning_rate, momentum):
    if optimizer_type == OptimizerTypes.asgd:
        return torch.optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.adam:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerTypes.sgd:
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
