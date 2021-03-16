import os
import sys
import networkx as nx
import numpy as np

from Components.RBC_REG.RBC import RBC
from Utils import CommonStr

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
        for g in self.g_lst:
            node2vec = Node2Vec(dimensions=self.dimensions)
            node2vec.fit(g)
            embedding = node2vec.get_embedding()
            embeddings_lst.append(embedding)
        return embeddings_lst

    def generate_samples(self, embedding, R):
        samples = []
        embeddings_Routings =  {}
        for s in range(0, len(R)):
            for t in range(0, len(R)):
                for u in range(0, len(R)):
                    for v in range(0, len(R)):
                        samples.append((torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                     dtype=torch.float, device=DEVICE), torch.tensor([R[s, t][v, u]],
                                                                                                     device=DEVICE,
                                                                                                     dtype=DTYPE)))
                        embeddings_Routings.update(
                            {f'{s}_{u}_{v}_{t}': (torch.tensor([embedding[s], embedding[u], embedding[v], embedding[t]],
                                                               dtype=torch.float), torch.tensor([R[s, t][v, u]], dtype=torch.float, device=DEVICE))})
        return samples, embeddings_Routings

    def generate_dataset(self, embeddings):
        samples = []
        rotuings = []
        for i in range(0, len(embeddings)):
            new_samples, new_rotuings = self.generate_samples(embeddings[i], self.r_lst[i])
            samples += new_samples
            rotuings.append(new_rotuings)
        return samples, rotuings

    def train_model(self, samples):
        self.model = self.e_ml.predict(samples, dimensions=self.dimensions)
        return self.model

    def predict_routing(self, embeddings):
        num_nodes = len(embeddings)
        predicted_R = torch.full(size=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0.0, dtype=torch.float,
                                 device=DEVICE)
        self.model.eval()
        for s in range(0, num_nodes):
            for t in range(0, num_nodes):
                for u in range(0, num_nodes):
                    for v in range(0, num_nodes):
                        predicted_R[s, t][v, u] = self.model(
                            torch.tensor([[embeddings[s], embeddings[u], embeddings[v], embeddings[t]]],
                                         dtype=torch.float, device=DEVICE))
        return predicted_R


if __name__ == '__main__':
    # union_graph = nx.disjoint_union_all(graphs)
    # df = pd.read_csv(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\Combined_Results\statistics.csv')
    paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\9_nodes\2_edges\4']
             # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\3_nodes\2_edges\9']
    test_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\3_nodes\2_edges\6']
    # paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\5',
    #          r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\6',
    #          r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\7',
    #          r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\8',
    #          r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\9',
    #          r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\10']
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\11',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\12']
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\13',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\15',
    # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\16']

    R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in paths]
    test_R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=DTYPE, device=DEVICE) for path in
                  test_paths]
    test_T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=DTYPE, device=DEVICE) for path in test_paths]
    test_G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in test_paths]

    # union_graph = nx.disjoint_union_all(G_lst)
    # G_lst = [union_graph]

    model_builder = EmbeddingModelBuilder(5, G_lst, R_lst, T_lst)
    all_embeddings = model_builder.compute_embeddings()
    sampels_x, rotuing_x = model_builder.generate_dataset(all_embeddings)
    trained_model = model_builder.train_model(sampels_x)

    model_builder_test = EmbeddingModelBuilder(5, test_G_lst, test_R_lst, test_T_lst)
    all_embeddings_test = model_builder_test.compute_embeddings()

    expected_rbc = RBC(eigenvector_method=CommonStr.EigenvectorMethod.power_iteration, pi_max_error=0.000001,
                       device=DEVICE, dtype=DTYPE).compute_rbc(G_lst[0], R_lst[0], T_lst[0])
    routing_prediction = model_builder.predict_routing(embeddings=all_embeddings_test[0])
    actual_rbc = RBC(eigenvector_method=CommonStr.EigenvectorMethod.power_iteration, pi_max_error=0.000001,
                     device=DEVICE, dtype=DTYPE).compute_rbc(test_G_lst[0], routing_prediction, test_T_lst[0])
    print(f'expected rbc: {expected_rbc}')
    print(f'actual rbc: {actual_rbc}')
