import torch
import math
from torch import nn
import networkx as nx
from Components import Policy
import numpy as np


def predict_degree(input_features, target_degree):
    dtype = torch.float
    device = torch.device("cuda:0")

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.tensor(input_features.clone().detach(), device=device, dtype=dtype)
    y = torch.tensor(target_degree.clone().detach(), device=device, dtype=dtype)

    # Create random Tensors for weights. we need len(input features) weights.
    weights = torch.randn(input_features.size()[0], input_features.size()[1], requires_grad=True, device=device, dtype=dtype)
    learning_rate = 0.00000001

    epoch = 0
    loss_val = 1000000000000
    # for e in range(epochs):
    while loss_val > 0.5:
        epoch += 1
        y_pred = torch.sum(x * weights, 0)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = torch.nn.MSELoss()
        d_loss = loss(y_pred, y)
        # if e % 10 == 0:
        loss_val = d_loss.item()
        print(epoch, loss_val)

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad .... will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d .... respectively.
        d_loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            weights -= learning_rate * weights.grad

    print(f'predicted degree is: {y_pred} actual degree is {y}')


if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # predict_degree(torch.tensor([[0, 0.5], [0, 0.5], [0, 0.5], [0, 0.5], [0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0]]),
    #                torch.tensor([1, 1]))

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
    input_features = torch.tensor(lst_inputs).transpose(0, 1)
    target_degree = torch.tensor([2, 2, 4, 1, 1])
    predict_degree(input_features, target_degree)
