import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.Node2Vec import Node2Vec
import torch

DEVICE = torch.device('cuda:0')
DTYPE = torch.float


class PreProcessor:
    def __init__(self, embeddings_dimensions, g_lst, r_lst, t_lst):
        self.dimensions = embeddings_dimensions
        self.g_lst = g_lst
        self.r_lst = r_lst
        self.t_lst = t_lst

    def pre_process_data(self, testing_mode=False):
        embeddings = self.compute_embeddings()
        samples = []
        routing = []
        for i in range(0, len(embeddings)):
            new_samples, new_routing = self.generate_samples(embeddings[i], self.r_lst[i])
            samples += new_samples
            routing.append(new_routing)

        return (samples, routing) if testing_mode else samples

    def compute_embeddings(self):
        embeddings_lst = []
        node2vec = Node2Vec(dimensions=self.dimensions)
        for g in self.g_lst:
            node2vec.fit(g)
            embedding = node2vec.get_embedding()
            embeddings_lst.append(embedding)
        return embeddings_lst

    def generate_samples(self, embedding, R):
        samples = []
        embeddings_routing = {}
        for s in range(0, len(R)):
            for t in range(0, len(R)):
                for u in range(0, len(R)):
                    for v in range(0, len(R)):
                        sample_features = torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                       device=DEVICE, dtype=DTYPE)
                        sample_label = torch.tensor([R[s, t][v, u]], device=DEVICE, dtype=DTYPE)
                        samples.append((sample_features, sample_label))
                        embeddings_routing.update({f'{s}_{u}_{v}_{t}': (sample_features, sample_label)})

        return samples, embeddings_routing



