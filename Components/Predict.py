import networkx as nx
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from Components import Policy
from Components import RBC




if __name__ == '__main__':

    edges = [('v0', 'v1'), ('v1', 'v2'), ('v1', 'v3'), ('v1', 'v4')]
    g3 = nx.Graph(edges)
    nodes_map_g3 = {k: v for v, k in enumerate(list(g3.nodes()))}
    degree_policy = Policy.DegreePolicy()
    R = degree_policy.get_policy_tensor(g3, nodes_map_g3)
    T = degree_policy.get_t_tensor(g3)
    R_flatten = torch.flatten(R, start_dim=1)
    y = torch.tensor([1, 4, 1, 1, 1])
    # rbc_tensor = RBC.rbc(g3, R, T)
    num_nodes_g3 = g3.number_of_nodes()




