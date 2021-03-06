import random

import networkx as nx
import matplotlib.pyplot as plt
from Components.Embedding.Node2Vec import Node2Vec
from Utils.GraphGenerator import GraphGenerator


if __name__ == '__main__':
    graphs = GraphGenerator.generate_rand_graphs()
    union_graph = nx.disjoint_union_all(graphs)
    cc = list(nx.connected_components(union_graph))
    rand_nodes_cc = [random.sample(nodes, 1)[0] for nodes in cc]
    rand_edges = [(rand_nodes_cc[len(rand_nodes_cc) - 1], rand_nodes_cc[0])]
    for i in range(0, len(rand_nodes_cc)-1):
        rand_edges.append((rand_nodes_cc[i], rand_nodes_cc[i+1]))
    union_graph.add_edges_from(rand_edges)
    node2vec = Node2Vec(dimensions=5)
    node2vec.fit(union_graph)
    embedding = node2vec.get_embedding()


