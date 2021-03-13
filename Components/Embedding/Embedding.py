import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from Components.Embedding.DeepWalk import DeepWalk
from Components.Embedding.NeuralNetwork import EmbeddingML
from Components.Embedding.Node2Vec import Node2Vec
from Components.RBC_REG.Policy import Policy, BetweennessPolicy
from Utils.GraphGenerator import GraphGenerator
import torch
#
# def normalize_embeddings(embeddings):
#    a sum(em)
def compute_embeddings(g, R, T, dimensions):
    node2vec = Node2Vec(dimensions=dimensions)
    node2vec.fit(g)
    embedding = node2vec.get_embedding()
    embedding = embedding - (sum(embedding) / 4)
    embeddings_lst = []
    embeddings_Routings = {}
    for s in range(0, len(R)):
        for t in range(0, len(R)):
            for u in range(0, len(R)):
                for v in range(0, len(R)):
                    embeddings_lst.append((torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                        dtype=torch.float, device=torch.device('cpu')), torch.tensor([R[s, t][v, u]],  device=torch.device('cpu'), dtype=torch.float)))
                    embeddings_Routings.update(
                        {f'{s}_{u}_{v}_{t}': (torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                           dtype=torch.float), torch.tensor([R[s, t][v, u]], dtype=torch.float, device=torch.device('cpu')))})

    features = torch.stack([embedding for embedding, label in embeddings_lst])
    labels = torch.stack([label for embedding, label in embeddings_lst])
    ml = EmbeddingML()
    model = ml.predict(features, labels, dimensions=dimensions)
    predicted_R = torch.full(size=(R.size()), fill_value=0.0, dtype=torch.float, device=torch.device('cpu'))
    for s in range(0, len(R)):
        for t in range(0, len(R)):
            for u in range(0, len(R)):
                for v in range(0, len(R)):
                    predicted_R[s, t][v, u] = model(torch.tensor([[embedding[s], embedding[u], embedding[v], embedding[t]]], dtype=torch.float))
    print(predicted_R)


if __name__ == '__main__':
    # graphs = GraphGenerator.generate_rand_graphs()
    # union_graph = nx.disjoint_union_all(graphs)
    # df = pd.read_csv(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\Combined_Results\statistics.csv')
    path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\Degree\13_nodes\2_edges\6'
    adj_matrix = np.load(path + '\\adj_mat.npy')
    g = nx.convert_matrix.from_numpy_matrix(adj_matrix)
    routing_policy = np.load(path + '\\routing_policy.npy')
    R = torch.tensor(routing_policy, dtype=torch.float)
    traffic_matrix = np.load(path + '\\traffic_mat.npy')
    T = torch.tensor(traffic_matrix, dtype=torch.float)
    # edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    # g = nx.Graph(edges)
    # nodes_mapping = {k: v for v, k in enumerate(list(g.nodes()))}
    # b_policy = BetweennessPolicy()
    # R = b_policy.get_policy_tensor(g, nodes_mapping=nodes_mapping)
    compute_embeddings(g, R, None, 5)



