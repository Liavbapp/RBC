import os
import random
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.Node2Vec import Node2Vec
import torch


class PreProcessor:

    def __init__(self, device, dtype, dimensions):
        self.device = device
        self.dtype = dtype
        self.dimensions = dimensions

    def generate_all_samples(self, embeddings, Rs, testing_mode=False):

        samples = []
        routing = []
        for i in range(0, len(embeddings)):
            new_samples, new_routing = self.generate_samples_for_graph(embeddings[i], Rs[i])
            samples += new_samples
            routing.append(new_routing)

        return (samples, routing) if testing_mode else samples

    def compute_embeddings(self, Gs):
        embeddings_lst = []
        total_sum = 0
        total_instances = 0
        list_arrays = []
        i = 0
        for g in Gs:
            node2vec = Node2Vec(dimensions=self.dimensions, seed=random.randrange(100))
            node2vec.fit(g)
            embedding = node2vec.get_embedding()
            total_sum += embedding.sum()
            total_instances += g.number_of_nodes() * self.dimensions
            list_arrays.append(embedding.flatten())
            embeddings_lst.append(embedding)
            i += 1
        total_avg = total_sum / total_instances
        union_arr = np.concatenate(list_arrays, axis=0)
        std_embedding = union_arr.std()
        embeddings_lst = [(embedding_i - total_avg) / std_embedding for embedding_i in embeddings_lst]
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



