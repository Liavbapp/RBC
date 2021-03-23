import torch
import torch.nn as nn

#
# class NeuralNetworkMatrix(nn.Module):
#     def __init__(self, dimensions, device, dtype):
#         super(NeuralNetworkMatrix, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.BatchNorm1d(dimensions * 2),
#             nn.Linear(dimensions * 3, 1000),
#             nn.BatchNorm1d(1000),
#             nn.LeakyReLU(),
#             nn.Linear(1000, 500),
#             nn.BatchNorm1d(500),
#             nn.LeakyReLU(),
#             nn.Linear(500, 300),
#             nn.BatchNorm1d(300),
#             nn.LeakyReLU(),
#             nn.Linear(300, 30),
#             nn.BatchNorm1d(30),
#             nn.LeakyReLU(),
#             nn.Linear(30, 1),
#             nn.Softmax()
#         ).to(device=device, dtype=dtype)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         prop_res = self.linear_relu_stack(x)
#         return prop_res
#
#
# class NeuralNetworkSoftMax(nn.Module):
#     def __init__(self, dimensions, device, dtype):
#         super(NeuralNetworkSoftMax, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.BatchNorm1d(dimensions * 3),
#             nn.Linear(dimensions * 3, 1000),
#             nn.BatchNorm1d(1000),
#             nn.LeakyReLU(),
#             nn.Linear(1000, 500),
#             nn.BatchNorm1d(500),
#             nn.LeakyReLU(),
#             nn.Linear(500, 300),
#             nn.BatchNorm1d(300),
#             nn.LeakyReLU(),
#             nn.Linear(300, 30),
#             nn.BatchNorm1d(30),
#             nn.LeakyReLU(),
#             nn.Linear(30, 1),
#             nn.Softmax()
#         ).to(device=device, dtype=dtype)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         prop_res = self.linear_relu_stack(x)
#         return prop_res


class NeuralNetwork(nn.Module):
    def __init__(self, dimensions, device, dtype):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(dimensions * 4),
            nn.Linear(dimensions * 4, 1000),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            # nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Linear(500, 300),
            # nn.BatchNorm1d(300),
            nn.LeakyReLU(),
            nn.Linear(300, 30),
            # nn.BatchNorm1d(30),
            nn.LeakyReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res
