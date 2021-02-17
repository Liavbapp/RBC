import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from Components.RBC_ML.RbcML import learn_models
import Components.RBC_REG.RBC as RBC_REG

DEVICE = torch.device('cuda:0')
DTYPE = torch.float


def test_degree_on_graph(g, adj_matrix, test_num):
    print(f'started testing figure_{test_num}')
    target_matrix = torch.sum(adj_matrix, dim=1).to(dtype=DTYPE, device=DEVICE)  # nodes degree matrix
    t_model, r_model = learn_models(adj_matrix, target_matrix, learning_rate=1e-2, epochs=100, momentum=0,
                                    optimizer_type="adam", max_error=0.00001)
    t_model = t_model.to(device=torch.device("cpu"))
    r_model = r_model.to(device=torch.device("cpu"))
    rbc_pred = RBC_REG.compute_rbc(g, r_model, t_model)
    print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')


def test_degree():
    edge_lst_minus1 = [(0, 1), (1, 2)]
    edge_lst_eas = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edge_lst_0 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (3, 4), (2, 4)]
    edge_lst_0_1 = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    edge_lst_1 = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 5)]
    edge_lst_2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
    edge_lst_3 = [(0, 1), (1, 2), (0, 3), (2, 3), (3, 4), (2, 4), (0, 4), (4, 5)]
    edge_lst_4 = [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4)]
    edge_lst_5 = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 5), (3, 6), (4, 5), (5, 6), (1, 4), (5, 7), (6, 7),
                  (4, 8), (7, 8), (8, 9), (4, 9)]
    all_edge_lst = [edge_lst_0]

    # i = 0
    # for edge_lst in all_edge_lst:
    #     i += 1
    #     g = nx.Graph(edge_lst)
    #     plt.figure(i)
    #     nx.draw(g, with_labels=True)
    # show(block=False)

    i = 0
    for edge_lst in all_edge_lst:
        i += 1
        g = nx.Graph(edge_lst)
        adj_matrix = torch.tensor(nx.adj_matrix(g).todense(), dtype=DTYPE)
        test_degree_on_graph(g, adj_matrix, test_num=i)


def draw_routing(routing_policy, s, t):
    routing_matrix_t = routing_policy[s, t].t()
    edges = [np.array(row.detach().to(device=torch.device('cpu'))) for row in list(torch.nonzero(routing_matrix_t))]
    edges_weights = [(i, j, {'weight': routing_matrix_t[i, j].item()}) for i, j in edges]
    g = nx.DiGraph(edges_weights)
    g.add_nodes_from(list(range(0, len(routing_matrix_t[0]))))
    # edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx(g, arrows=True)
    show()


if __name__ == '__main__':
    test_degree()
