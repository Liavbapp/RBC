import os
import random
from Utils.Auxaliry import create_uv_matrix_ordered, create_uv_matrix_combined
import sys
from itertools import product

import numpy as np
import networkx as nx
from Utils.CommonStr import NumRandomSamples

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))

import torch


class PreProcessor:

    def __init__(self, device, dtype, dim):
        self.device = device
        self.dtype = dtype
        self.dimensions = dim

    def generate_samples_to_st_routing_optim(self, embeddings, Gs, Rs):

        lst_samples = []
        i = 0
        for embedding, R in zip(embeddings, Rs):
            i += 1
            num_nodes = len(R)

            embedding = torch.from_numpy(embedding).to(device=self.device, dtype=self.dtype)
            dim = embedding.shape[1]
            uv_tensor = create_uv_matrix_ordered(embedding, self.device)
            rand_uv_idx = torch.randint(low=0, high=num_nodes ** 2, size=(num_nodes,), device=self.device)
            uv_rand = torch.index_select(uv_tensor, 0, rand_uv_idx)

            st_tuples = list(product(range(num_nodes), range(num_nodes)))
            s_lst, t_lst = zip(*st_tuples)

            def gen_st_samples(s, t):
                emb_s, emb_t = embedding[s].repeat(repeats=(num_nodes, 1)), embedding[t].repeat(repeats=(num_nodes, 1))
                R_st = torch.index_select(torch.flatten(R[0, 0]).unsqueeze(dim=1), 0, rand_uv_idx)
                sample = torch.cat([emb_s, uv_rand, emb_t, R_st], dim=1)
                return sample

            samples = list(map(gen_st_samples, s_lst, t_lst))
            lst_samples.append(torch.cat(samples, dim=0))

        # samples = torch.cat(lst_samples, dim=0)
        samples = torch.stack(lst_samples)
        return samples

    def generate_samples_to_centrality_optim(self, embeddings, Ts, Gs, Rbcs):

        nodes_embed_lst, graphs_embed_lst = [], []
        mult_const = [torch.tensor(nx.to_numpy_matrix(G), device=self.device, dtype=self.dtype).fill_diagonal_(0) for G
                      in Gs]
        add_const = [torch.zeros(size=(G.number_of_nodes(), G.number_of_nodes()), device=self.device,
                                 dtype=self.dtype).fill_diagonal_(1) for G in Gs]

        for node_embedding, graph_embedding in embeddings:
            nodes_embed_lst.append(torch.tensor(node_embedding, device=self.device, dtype=self.dtype))
            graphs_embed_lst.append(torch.tensor(graph_embedding, device=self.device, dtype=self.dtype))

        return list(
            map(lambda lst: torch.stack(lst), [nodes_embed_lst, graphs_embed_lst, Ts, mult_const, add_const, Rbcs]))

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

    def generate_all_samples(self, embeddings, Rs, testing_mode=False, path=None):
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
        samples = torch.stack([sample for sample in samples])
        return samples

    def compute_embeddings_for_centrality_optim(self, Gs, seeds, embedding_alg_nodes, embedding_alg_graph):
        nodes_embeddings = self.compute_node_embeddings(Gs, seeds, embedding_alg_nodes)
        graphs_embeddings = self.compute_graphs_embeddings(Gs, seeds, embedding_alg_graph)
        return list(zip(nodes_embeddings, graphs_embeddings))

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
            sample_features = torch.stack([embd_torch[s], embd_torch[u], embd_torch[v], embd_torch[t]]).flatten()
            sample_label = torch.stack([R[s, t][v, u]])
            samples.append(torch.cat((sample_features, sample_label), dim=0))
            # samples.append((sample_features, sample_label))
        return samples
