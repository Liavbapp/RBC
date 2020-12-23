import torch
import math
from torch import nn
import networkx as nx
from Components import Policy
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


#

class DegreePrediction(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()
        device = torch.device('cuda:0')
        dtype = torch.float
        self.weights = torch.nn.Parameter(
            torch.randn(input_features.size()[0], input_features.size()[1], requires_grad=True, device=device,
                        dtype=dtype))

    def forward(self, x):
        return torch.sum(x * self.weights, dim=0)


def predit_degree_custom_model(x, y):
    model = DegreePrediction(input_features)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(2000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'prediction : {y_pred}, target:{y}')


#
# def predict_degree(input_features, target_degree):
#     dtype = torch.float
#     device = torch.device("cuda:0")
#
#     # Create Tensors to hold input and outputs.
#     # By default, requires_grad=False, which indicates that we do not need to
#     # compute gradients with respect to these Tensors during the backward pass.
#     # x = torch.tensor(input_features.clone().detach(), device=device, dtype=dtype)
#     # x = torch.tensor([[0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]], device=device, dtype=dtype)
#     # y = torch.tensor(target_degree.clone().detach(), device=device, dtype=dtype)
#     x = input_features
#     y = target_degree
#
#     # Create random Tensors for weights. we need len(input features) weights.
#     # weights = torch.randn(input_features.size()[0], input_features.size()[1], requires_grad=True, device=device, dtype=dtype)
#     # learning_rate = 0.0000001
#     model = torch.nn.Sequential(
#         torch.nn.Linear(x.size()[1], y.size()[0], bias=False),
#         torch.nn.Flatten(0, 1)
#     ).to(device=device, dtype=dtype, non_blocking=True)
#
#     learning_rate = 1e-3
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     epoch = 1500
#     loss_val = 1000000000000
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#
#     for e in range(epoch):
#         # while loss_val > 0.5:
#         epoch += 1
#         # y_pred = torch.sum(x * weights, 0)
#         y_pred = model(x)
#         # Compute and print loss using operations on Tensors.
#         # Now loss is a Tensor of shape (1,)
#         # loss.item() gets the scalar value held in the loss.
#         loss = loss_fn(y_pred, y)
#         # if e % 10 == 0:
#         loss_val = loss.item()
#         print(e, loss_val)
#
#         # Zero the gradients before running the backward pass.
#         # model.zero_grad()
#         optimizer.zero_grad()
#         # Use autograd to compute the backward pass. This call will compute the
#         # gradient of loss with respect to all Tensors with requires_grad=True.
#         # After this call a.grad, b.grad. c.grad and d.grad .... will be Tensors holding
#         # the gradient of the loss with respect to a, b, c, d .... respectively.
#         loss.backward()
#
#         # Manually update weights using gradient descent. Wrap in torch.no_grad()
#         # because weights have requires_grad=True, but we don't need to track this
#         # in autograd.
#         # with torch.no_grad():
#         #     weights -= learning_rate * weights.grad
#         # with torch.no_grad():
#         #     for param in model.parameters():
#         #         param -= learning_rate * param.grad
#         optimizer.step()
#
#     print(f'predicted degree is: {y_pred} actual degree is {y}')


if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # predit_degree_custom_model(
    #     torch.tensor([[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5], [0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0]],
    #                  dtype=DTYPE, device=DEVICE),
    #     torch.tensor([1, 1], device=DEVICE, dtype=DTYPE))

    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1'), ('v0', 'v2'), ('v1', 'v2'), ('v2', 'v3')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # lst_inputs = []
    # for i in range(0, len(t_tensor)):
    #     row_i = t_tensor[i]
    #     row_i_inputs = []
    #     for j in range(0, len(row_i)):
    #         row_i_inputs += [row_i[j]] * (len(t_tensor) * len(t_tensor))
    #     lst_inputs.append(row_i_inputs)
    # input_features = torch.flatten(torch.tensor(lst_inputs, dtype=DTYPE, device=DEVICE))
    # input_features = input_features.view(1, input_features.size()[0])
    # target_degree = torch.tensor([2, 2, 3, 1], dtype=DTYPE, device=DEVICE)
    # predict_degree(input_features, target_degree)

    degree_policy = Policy.DegreePolicy()
    edge_lst = [('v0', 'v1'), ('v0', 'v2'), ('v1', 'v2'), ('v2', 'v3'), ('v2', 'v4')]
    g = nx.Graph(edge_lst).to_directed()
    t_tensor = degree_policy.get_t_tensor(g)
    lst_inputs = []
    for i in range(0, len(t_tensor)):
        row_i = t_tensor[i]
        row_i_inputs = []
        for j in range(0, len(row_i)):
            row_i_inputs += [row_i[j]] * (len(t_tensor) * len(t_tensor))
        lst_inputs.append(row_i_inputs)
    input_features = torch.tensor(lst_inputs, device=DEVICE, dtype=DTYPE).transpose(0, 1)
    target_degree = torch.tensor([2, 2, 4, 1, 1], device=DEVICE, dtype=DTYPE)
    predit_degree_custom_model(input_features, target_degree)
