import datetime

import torch
import numpy as np
import networkx as nx

from Components.Embedding.EmbeddingML import EmbeddingML
from Components.Embedding.PreProcessor import PreProcessor
from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import EigenvectorMethod

DTYPE = torch.float
DEVICE = torch.device('cuda:0')


def get_paths():
    train_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\8',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\7',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\9',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\10',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\11',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\12',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\13',
                   # r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\15',
                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\16']
    test_paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\5_nodes\2_edges\14']
    return train_paths, test_paths


def extract_info_from_path(paths):
    R_lst = [torch.tensor(np.load(path + '\\routing_policy.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    T_lst = [torch.tensor(np.load(path + '\\traffic_mat.npy'), dtype=DTYPE, device=DEVICE) for path in paths]
    G_lst = [nx.convert_matrix.from_numpy_matrix(np.load(path + '\\adj_mat.npy')) for path in paths]

    return R_lst, T_lst, G_lst


def test_embeddings():
    train_paths, tests_paths = get_paths()
    train_R_lst, train_T_lst, train_G_lst = extract_info_from_path(train_paths)
    test_R_lst, test_T_lst, test_G_lst = extract_info_from_path(tests_paths)

    embedding_dim = 5
    embedding_ml = EmbeddingML()
    train_pre_processor = PreProcessor(embeddings_dimensions=embedding_dim, g_lst=train_G_lst, r_lst=train_R_lst,
                                       t_lst=train_T_lst)
    samples = train_pre_processor.pre_process_data()
    start_time = datetime.datetime.now()
    model = embedding_ml.train_model(samples, embedding_dim)
    train_time = datetime.datetime.now() - start_time
    print(f'train time: {train_time}')

    test_preprocessor = PreProcessor(5, test_G_lst, test_R_lst, test_T_lst)

    rbc_train = RBC(EigenvectorMethod.power_iteration, pi_max_error=0.00001, device=DEVICE, dtype=DTYPE)
    rbc_test = RBC(EigenvectorMethod.power_iteration, pi_max_error=0.00001, device=DEVICE, dtype=DTYPE)
    expected_rbc = rbc_train.compute_rbc(train_G_lst[0], train_R_lst[0], train_T_lst[0])
    embeddings_test = test_preprocessor.compute_embeddings()[0]
    actual_rbc = rbc_test.compute_rbc(test_G_lst[0], embedding_ml.predict_routing(model, embeddings_test), train_T_lst[0])

    print(f'expected rbc: {expected_rbc}')
    print(f'actual rbc: {actual_rbc}')


if __name__ == '__main__':
    test_embeddings()
