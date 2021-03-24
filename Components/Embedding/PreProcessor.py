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

    def compute_embeddings(self, Gs):
        embeddings_lst = []
        row_sums = np.zeros(self.dimensions)
        dim_num_instances = np.zeros(self.dimensions)
        list_arrays = []
        i = 0
        for g in Gs:
            node2vec = Node2Vec(dimensions=self.dimensions, seed=random.randrange(100))
            node2vec.fit(g)
            embedding = node2vec.get_embedding()
            row_sums += embedding.sum(axis=0)
            dim_num_instances += g.number_of_nodes()
            list_arrays.append(embedding.flatten())
            embeddings_lst.append(embedding)
            i += 1
        avgs = row_sums / dim_num_instances
        stds = np.concatenate(np.stack(embeddings_lst, axis=1)).std(axis=0)

        # union_arr = np.concatenate(list_arrays, axis=0)
        # std_embedding = union_arr.std()
        # embeddings_lst = [(embedding_i - total_avg) / std_embedding for embedding_i in embeddings_lst]
        embeddings_lst = [(embedding_i - avgs) / stds for embedding_i in embeddings_lst]
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
