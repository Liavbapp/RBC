import torch
import math
from torch import nn
import networkx as nx
from Components import Policy
import numpy as np



def predict_degree(input_features, traffic_matrix, target_degree):
    # initial_weights = torch.rand(size=(8, 1))
    # output = torch.mm(input_features, initial_weights)
    # output.requires_grad = True


if __name__ == '__main__':
    # degree_policy = Policy.DegreePolicy()
    # edge_lst = [('v0', 'v1')]
    # g = nx.Graph(edge_lst).to_directed()
    # t_tensor = degree_policy.get_t_tensor(g)
    # predict_degree(torch.tensor([[0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]]), traffic_matrix=t_tensor,
    #                target_degree=torch.tensor([[1.0]]))

    dtype = torch.float
    device = torch.device("cuda:0")

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], dtype=float)
    y = torch.tensor(1.0, device=device, dtype=dtype)

    # Create random Tensors for weights. we need 8 weights: a,b,c,d,e,f,g,h
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    e = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    f = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    g = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    h = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    weights = np.array([[a], [b], [c], [d], [e], [f], [g], [h]])

    learning_rate = 1e-6

    for t in range(2000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = np.matmul(x, weights)[0]

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss_func = nn.MSELoss()
        delta_loss = loss_func(y_pred, y)
        if t % 100 == 99:
            print(t, delta_loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        delta_loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        grads = np.full(size=(len(x), 1), dtype=float, fill_value=0.0)
        for i in range(0, weights.size):
            grads[i][0] = weights[i][0].grad

        with torch.no_grad():
            weights -= learning_rate * grads

            # Manually zero the gradients after updating weights
            for i in range(0, weights.size):
                weights[i][0].grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')





