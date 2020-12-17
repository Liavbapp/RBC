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
    x = np.array(input_features, dtype=float)
    y = torch.tensor(target_degree, device=device, dtype=dtype)

    # Create random Tensors for weights. we need len(input features) weights.
    weights_lst = [[torch.randn((), device=device, dtype=dtype, requires_grad=True)] for k in range(input_features.shape[1])]

    weights = np.array(weights_lst)

    learning_rate = 0.01

    epochs = 1000
    for e in range(epochs):
        y_pred = np.matmul(x, weights)[0][0]

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = nn.MSELoss()
        d_loss = loss(y_pred, y)
        if e % 100 == 99:
            print(e, d_loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad .... will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d .... respectively.
        d_loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            # weights -= learning_rate * grads
            for weight in weights:
                weight[0] -= learning_rate * weight[0].grad
                # zero the grad manually
                weight[0].grad = None

    print(f'degree is: {y_pred}')



if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    predict_degree(np.array([[0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]], dtype=float), target_degree=1)
