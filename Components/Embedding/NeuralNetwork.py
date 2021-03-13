import datetime
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from Components.Embedding.Node2Vec import Node2Vec
from Utils.GraphGenerator import GraphGenerator


class NeuralNetwork(nn.Module):
    def __init__(self, dimensions):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dimensions * 4, dimensions * 4),
            nn.BatchNorm1d(dimensions * 4),
            nn.ReLU(),
            nn.Linear(dimensions * 4, dimensions * 1),
            nn.BatchNorm1d(dimensions * 1),
            nn.ReLU(),
            nn.Linear(dimensions * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class EmbeddingML:

    def predict(self, features, labels, dimensions):
        start_time = datetime.datetime.now()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        # expected_res = torch.tensor([0.7], dtype=torch.float)
        learning_rate = 1e-4
        model = NeuralNetwork(dimensions)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        i = 0
        for t in range(0, 500000):
            y_pred = model(features)
            i += 1
            loss = loss_fn(y_pred, labels)
            print(t, loss.item()) if t % 200 == 0 else 1
            optimizer.zero_grad()
            loss.backward()  # backward unable handle 50 nodes
            optimizer.step()

        return model


if __name__ == '__main__':
    pass

    # graphs = GraphGenerator.generate_rand_graphs()
    # union_graph = nx.disjoint_union_all(graphs)
    # cc = list(nx.connected_components(union_graph))
    # rand_nodes_cc = [random.sample(nodes, 1)[0] for nodes in cc]
    # rand_edges = [(rand_nodes_cc[len(rand_nodes_cc) - 1], rand_nodes_cc[0])]
    # # for i in range(0, len(rand_nodes_cc)-1):
    # #     rand_edges.append((rand_nodes_cc[i], rand_nodes_cc[i+1]))
    # # union_graph.add_edges_from(rand_edges)
    # node2vec = Node2Vec(dimensions=5)
    # node2vec.fit(union_graph)
    # embedding = node2vec.get_embedding()
    # x = embedding[0]
    # ml = EmbeddingML()
    # ml.predict(x)
    # a = 1
