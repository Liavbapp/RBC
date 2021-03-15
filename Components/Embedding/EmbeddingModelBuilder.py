import os
import sys
import networkx as nx
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
from Components.Embedding.NeuralNetwork import EmbeddingML
from Components.Embedding.Node2Vec import Node2Vec
import torch

DEVICE = torch.device('cpu')
DTYPE = torch.float


class EmbeddingModelBuilder:
    def __init__(self, dimensions, g_lst, r_lst, t_lst):
        self.e_ml = EmbeddingML()
        self.dimensions = dimensions
        self.g_lst = g_lst
        self.r_lst = r_lst
        self.t_lst = t_lst

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
        for s in range(0, len(R)):
            for t in range(0, len(R)):
                for u in range(0, len(R)):
                    for v in range(0, len(R)):
                        samples.append((torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                     dtype=torch.float, device=DEVICE), torch.tensor([R[s, t][v, u]],
                                                                                                     device=DEVICE,
                                                                                                     dtype=DTYPE)))
                        # embeddings_Routings.update(
                        #     {f'{s}_{u}_{v}_{t}': (torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                        #                                        dtype=torch.float), torch.tensor([R[s, t][v, u]], dtype=torch.float, device=DEVICE))})
        return samples

    def generate_dataset(self, embeddings):
        samples = []
        for i in range(0, len(embeddings)):
            samples += self.generate_samples(embeddings[i], self.r_lst[i])

        features = torch.stack([embedding for embedding, label in samples])
        labels = torch.stack([label for embedding, label in samples])

        return features, labels

    def train_model(self, features, labels):
        model = self.e_ml.predict(features, labels, dimensions=self.dimensions)
        return model

    # def check_model(self):
    #     predicted_R = torch.full(size=(R.size()), fill_value=0.0, dtype=torch.float, device=DEVICE)
    #     for s in range(0, len(R)):
    #         for t in range(0, len(R)):
    #             for u in range(0, len(R)):
    #                 for v in range(0, len(R)):
    #                     predicted_R[s, t][v, u] = model(
    #                         torch.tensor([[embedding[s], embedding[u], embedding[v], embedding[t]]], dtype=torch.float))
    #     print(predicted_R)


if __name__ == '__main__':
    # graphs = GraphGenerator.generate_rand_graphs()
    # union_graph = nx.disjoint_union_all(graphs)
    # df = pd.read_csv(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\Combined_Results\statistics.csv')
    paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\5',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\6',
             # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\7',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\8',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\9',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\10',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\11',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\12',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\13',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\15',
             r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\16']

    R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in paths]

    model_builder = EmbeddingModelBuilder(5, G_lst, R_lst, T_lst)
    all_embeddings = model_builder.compute_embeddings()
    features_x, labels_y = model_builder.generate_dataset(all_embeddings)
    model = model_builder.train_model(features_x, labels_y)
    a


