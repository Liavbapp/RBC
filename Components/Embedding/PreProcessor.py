import os
import random
import sys

import networkx as nx
import numpy as np
from karateclub import Diff2Vec, NodeSketch, NNSED, RandNE, GLEE, MNMF, SocioDim, NetMF, LaplacianEigenmaps

from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import EigenvectorMethod, NumRandomSamples

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))

import torch


class PreProcessor:

    def __init__(self, device, dtype, dim):
        self.device = device
        self.dtype = dtype
        self.dimensions = dim

    def generate_all_samples_embeddings_to_rbc(self, embeddings, Rs, testing_mode=False):

        return [(torch.tensor(embedding, device=self.device, dtype=self.dtype), graph, R, T, rbc) for
                embedding, graph, R, T, rbc in embeddings]

    def generate_all_samples_embeddings_to_routing(self, embeddings, Rs, testing_mode=False):
        samples = []
        routing = []
        for i in range(0, len(embeddings)):
            samples.append((torch.tensor(embeddings[i], device=self.device, dtype=self.dtype), Rs[i].view(-1)))
            routing.append(Rs[i])

        return (samples, routing) if testing_mode else samples

    def generate_all_samples_s_t_routing(self, embeddings, Rs, testing_mode=False):
        samples = []
        routing = []
        for i in range(0, len(embeddings)):
            new_samples, new_routing = self.generate_samples_for_st_routing(embeddings[i], Rs[i])
            samples += new_samples
            routing.append(new_routing)
        return samples

    def generate_all_samples(self, embeddings, Rs, testing_mode=False , path=None):
        if path is not None:
            samples = self.load_samples(path)
            return samples
        samples = []
        routing = []
        num_embeddings = len(embeddings)
        for i in range(0, num_embeddings):
            new_samples, new_routing = self.generate_samples_for_graph(embeddings[i], Rs[i])
            samples += new_samples
            routing.append(new_routing)

        return (samples, routing) if testing_mode else samples

    def generate_random_samples(self, embeddings, Rs, num_rand_samples):
        samples = []
        num_embeddings = len(embeddings)
        for i in range(0, num_embeddings):
            new_samples = self.generate_random_samples_for_graph(embeddings[i], Rs[i], num_rand_samples)
            samples += new_samples
        return samples


    def compute_graphs_embeddings(self, Gs, seeds, embedding_alg):
        embedding_alg.fit(Gs)
        embeddings_lst = embedding_alg.get_embedding()
        return embeddings_lst

    def compute_node_embeddings(self, Gs, seeds, embedding_alg):
        embeddings_lst = []
        for g, seed in zip(Gs, seeds):
            embedding_alg.seed = seed
            embedding_alg.fit(g)
            embedding = embedding_alg.get_embedding()
            embeddings_lst.append(embedding)
        return embeddings_lst

    def generate_samples_for_st_routing(self, embedding, R):
        samples = []
        embedding_routing = {}
        for s in range(0, len(R)):
            for t in range(0, len(R)):
                samples_features = torch.tensor([embedding[s], embedding[t]], device=self.device, dtype=self.dtype)
                sample_label = torch.tensor(R[s, t], device=self.device, dtype=self.dtype)
                samples.append((samples_features, sample_label))
                embedding_routing.update({f'{s}_{t}': (samples_features, sample_label)})

        return samples, embedding_routing

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
                        # embeddings_routing.update({f'{s}_{u}_{v}_{t}': (sample_features, sample_label)})

        return samples, embeddings_routing

    def generate_random_samples_for_graph(self, embedding, R, num_rand_samples):
        samples = []
        embd_torch = torch.tensor(embedding, device=self.device, dtype=self.dtype)
        num_nodes = len(R)
        if num_rand_samples == NumRandomSamples.N_power_2:
            num_samples = num_nodes ** 2
        if num_rand_samples == NumRandomSamples.N:
            num_samples = num_nodes

        for i in range(0, num_samples):
                s, u, v, t = np.random.choice(range(num_nodes), 4)
                sample_features = torch.stack([embd_torch[s], embd_torch[u], embd_torch[v], embd_torch[t]])
                sample_label = torch.stack([R[s, t][v, u]])
                samples.append((sample_features, sample_label))
        return samples



    def save_samples(self, samples):
        a = torch.stack([sample[0] for sample in samples])
        b = torch.stack([sample[1] for sample in samples])
        torch.save(a, r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\Data\features.pt')
        torch.save(b, r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\Data\labels.pt')

    def load_samples(self, path):
        features = torch.load(f'{path}\\features.pt')
        labels = torch.load(f'{path}\\labels.pt')
        features = list(torch.unbind(features))
        labels = list(torch.unbind(labels))
        return list(zip(features, labels))
