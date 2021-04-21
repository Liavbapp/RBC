import torch
import torch.nn as nn

from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import EigenvectorMethod


class NeuralNetworkNodeEmbeddingSourceTargetRouting(nn.Module):
        def __init__(self, dimensions, num_nodes, device, dtype):
            super(NeuralNetworkNodeEmbeddingSourceTargetRouting, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(dimensions * 2, 1000),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, num_nodes ** 2),
                # nn.Sigmoid()
            ).to(device=device, dtype=dtype)

        def forward(self, x):
            x = self.flatten(x)
            prop_res = self.linear_relu_stack(x)
            return prop_res


class NeuralNetworkGraphEmbeddingRbc(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkGraphEmbeddingRbc, self).__init__()
        self.num_nodes = num_nodes
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(dimensions),
            nn.Linear(dimensions, 50),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            # nn.Linear(100, 100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            nn.Linear(50, 50),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(50, num_nodes ** 4),
            nn.Sigmoid(),
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)

        return prop_res.view(prop_res.shape[0], self.num_nodes, self.num_nodes, self.num_nodes, self.num_nodes)



class NeuralNetworkGraphEmbeddingRouting(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkGraphEmbeddingRouting, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(dimensions),
            nn.Linear(dimensions, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, num_nodes ** 4),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res


class NeuralNetworkNodesEmbeddingRouting(nn.Module):
    def __init__(self, dimensions, num_nodes, device, dtype):
        super(NeuralNetworkNodesEmbeddingRouting, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.BatchNorm1d(num_nodes * dimensions),
            nn.Linear(num_nodes * dimensions, 1000),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 200),
            # nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 5000),
            # nn.BatchNorm1d(5000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(5000, 1000),
            # nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, num_nodes ** 4),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res


class NeuralNetwork(nn.Module):
    def __init__(self, dimensions, device, dtype):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(dimensions * 4),
            nn.Linear(dimensions * 4, 1000),
            nn.ReLU(),
            nn.Linear(1000, 900),
            nn.ReLU(),
            nn.Linear(900, 800),
            nn.ReLU(),
            nn.Linear(800, 700),
            nn.ReLU(),
            nn.Linear(700, 600),
            nn.ReLU(),
            nn.Linear(600, 500),
            nn.ReLU(),
            nn.Linear(500, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            # nn.Sigmoid()
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        x = self.flatten(x)
        prop_res = self.linear_relu_stack(x)
        return prop_res
