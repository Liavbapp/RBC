import networkx as nx
import numpy as np
import torch
import torchvision
from time import time
import random
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torchvision import datasets, transforms
from torch import nn, optim
from Components import Policy
import Components.RBC as RBC
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Components.Dataset import MyOwnDataset
from Components.Net import Net


def approximate_degree_centrality(g):
    nodes_map_g3 = {k: v for v, k in enumerate(list(g3.nodes()))}
    R, T = get_policies(g3, nodes_map_g3)
    data = get_data(R, T, g3, nodes_map_g3)
    a = 1


def pre_processing():
    data_set = MyOwnDataset(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset')
    data_set = data_set.shuffle()
    train_set = data_set[:2]
    validation_set = data_set[2:3]
    test_set = data_set[3:]
    return train_set, validation_set, test_set


if __name__ == '__main__':
    g_nisuy = nx.read_edgelist(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset\test.edgelist', create_using=nx.DiGraph)
    deg_policy = Policy.DegreePolicy()
    node_map = {k: v for v, k in enumerate(list(g_nisuy.nodes()))}
    R = deg_policy.get_policy_tensor(g_nisuy, node_map)
    T = deg_policy.get_t_tensor(g_nisuy)
    R_flat = torch.flatten(R, start_dim=1)
    T_flat = torch.flatten(T, start_dim=1)
    data_set = MyOwnDataset(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset')
    data = data_set[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data_set, R_flat, T_flat).to(device)
    data = data_set[0].to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    num_epocs = 200
    for epoch in range(num_epocs):
        print(f'{epoch} out of {num_epocs}')
        optimzer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(loss)
        loss.backward()
        optimzer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / sum(data.test_mask)
    print('Accuracy: {:.4f}'.format(acc))
