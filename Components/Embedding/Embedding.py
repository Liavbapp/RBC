import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from Components.Embedding.DeepWalk import DeepWalk
from Components.Embedding.NeuralNetwork import EmbeddingML
from Components.Embedding.Node2Vec import Node2Vec
from Components.RBC_REG.Policy import Policy, BetweennessPolicy
from Utils.GraphGenerator import GraphGenerator
import torch

if __name__ == '__main__':
    # graphs = GraphGenerator.generate_rand_graphs()
    # union_graph = nx.disjoint_union_all(graphs)
    # # cc = list(nx.connected_components(union_graph))
    # # rand_nodes_cc = [random.sample(nodes, 1)[0] for nodes in cc]
    # # rand_edges = [(rand_nodes_cc[len(rand_nodes_cc) - 1], rand_nodes_cc[0])]
    # # for i in range(0, len(rand_nodes_cc)-1):
    # #     rand_edges.append((rand_nodes_cc[i], rand_nodes_cc[i+1]))
    # # union_graph.add_edges_from(rand_edges)
    # node2vec = Node2Vec(dimensions=5)
    # node2vec.fit(union_graph)
    # embedding = node2vec.get_embedding()
    # df = pd.DataFrame(data=embedding)
    # a = 1
    # fig = px.scatter(df, x="x", y="y", size_max=60)
    # fig.show()
    edges = [(0, 1), (0, 2), (1, 2)]
    g = nx.Graph(edges)
    b_policy = BetweennessPolicy()
    R = b_policy.get_policy_tensor(g, nodes_mapping={k: v for v, k in enumerate(list(g.nodes()))})
    node2vec = Node2Vec(dimensions=5)
    node2vec.fit(g)
    embedding = node2vec.get_embedding()
    embeddings_lst = []
    embeddings_Routings = {}
    for s in range(0, len(R)):
        for t in range(0, len(R)):
            for u in range(0, len(R)):
                for v in range(0, len(R)):
                    embeddings_lst.append((torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                        dtype=torch.float), torch.tensor([R[s, t][v, u]])))
                    embeddings_Routings.update(
                        {f'{s}_{t}_{u}_{v}': (torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                           dtype=torch.float), torch.tensor([R[s, t][v, u]]))})

    features = torch.stack([embedding for embedding, label in embeddings_lst])
    labels = torch.stack([label for embedding, label in embeddings_lst])
    ml = EmbeddingML()
    model = ml.predict(features, labels)
    nisuy_res = model(torch.tensor([[embedding[0], embedding[0], embedding[2], embedding[2]]], dtype=torch.float))
    print(nisuy_res)