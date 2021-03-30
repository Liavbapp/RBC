import os
import random
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.Node2Vec import Node2Vec
import torch


class PreProcessor:

    def __init__(self, device, dtype, dim):
        self.device = device
        self.dtype = dtype
        self.dimensions = dim

    def generate_all_samples(self, embeddings, Rs, testing_mode=False):

        samples = []
        routing = []
        for i in range(0, len(embeddings)):
            new_samples, new_routing = self.generate_samples_for_graph(embeddings[i], Rs[i])
            samples += new_samples
            routing.append(new_routing)

        return (samples, routing) if testing_mode else samples

    def compute_embeddings(self, Gs, seeds):
        embeddings_lst = []

        for g, seed in zip(Gs, seeds):
            node2vec = Node2Vec(dimensions=self.dimensions, seed=seed)
            node2vec.fit(g)
            embedding = node2vec.get_embedding()
            embeddings_lst.append(embedding)
        return embeddings_lst

    def generate_samples_for_graph(self, embedding, R):
        samples = []
        embeddings_routing = {}
        for s in range(0, len(R)):
            for t in range(0, len(R)):
                for u in range(0, len(R)):
                    for v in range(0, len(R)):
                        sample_features = torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                       device=self.device, dtype=self.dtype)
                        sample_label = torch.tensor([R[s, t][v, u]], device=self.device, dtype=self.dtype)
                        samples.append((sample_features, sample_label))
                        embeddings_routing.update({f'{s}_{u}_{v}_{t}': (sample_features, sample_label)})

        return samples, embeddings_routing
