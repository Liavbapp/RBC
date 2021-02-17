import datetime
import torch
from Components.RBC_ML.RbcNetwork import RbcNetwork

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


def learn_models(adj_mat, y, hyper_params):
    zero_mat, const_mat = get_fixed_mat(adj_mat)
    model = RbcNetwork(adj_mat)
    criterion = torch.nn.MSELoss(reduction='sum')

    optimizer_type = hyper_params['optimizer_type']
    if optimizer_type == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=hyper_params['learning_rate'])
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params['learning_rate'],
                                    momentum=hyper_params['momentum'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params['learning_rate'],
                                    momentum=hyper_params['momentum'])

    start_time = datetime.datetime.now()
    for t in range(hyper_params['epochs']):
        y_pred = model(adj_mat, zero_mat, const_mat, hyper_params['pi_max_err'])
        loss = criterion(y_pred, y)
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # model.weights_t.data.clamp_(0)
        # model.weights_r.data.clamp_(min=0, max=1)

    print(f'\nRun Time -  {datetime.datetime.now() - start_time} \n\nLearning Target - {y}')
    model_t = model.weights_t
    model_r = torch.sigmoid(model.weights_r * zero_mat + const_mat)
    return model_t, model_r


def get_fixed_mat(adj_mat):
    adj_matrix_t = adj_mat.t()  # transpose the matrix to because R policy matrices are transposed (predecessor matrix)
    adj_size = adj_mat.size()[0]
    zeros_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    constant_mat = torch.full(size=(adj_size, adj_size, adj_size, adj_size), fill_value=0, dtype=DTYPE, device=DEVICE)
    for s in range(0, adj_size):
        for t in range(0, adj_size):
            constant_mat[s, t, s, s] = 1
            zeros_mat[s, t] = adj_matrix_t  # putting zeros where the transposed adj_matrix has zeros
            zeros_mat[s, t, s] = 0 # not works with both adjusments below
            zeros_mat[s, t, :, t] = 0
    return zeros_mat, constant_mat
