import networkx as nx
import plotly.express as px
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from Components.Embedding_Algorithms.DeepWalk import DeepWalk
from Components.Embedding_Algorithms.GraphWave import GraphWave
from Components.Embedding_Algorithms.Mause import MUSAE
from Components.Embedding_Algorithms.Node2Vec import Node2Vec
from Components.Embedding_Algorithms.Role2Vec import Role2Vec
from Components.Embedding_Algorithms.GL2VEC import GL2Vec
from Utils.CommonStr import StatisticsParams
from Utils.GraphGenerator import GraphGenerator


def plot_orig_graph(g, i):
    plt.figure(0)
    nx.draw(g, with_labels=True)


def draw_embeddings():
    gs = GraphGenerator('SPBC').graphs_for_embeddings_show()

    node2vec = Node2Vec(dimensions=2, p=np.inf, q=1/1999999999999999999999999999999999999999999999999999999999999999)
    # graph_wave = GraphWave(dimensions=2)
    for i in range(0, len(gs)):
        plot_orig_graph(gs[i], i)
        embeddings_lst = []
        for seed in range(0, 4):
            node2vec.seed = seed
            node2vec.fit(gs[i])
            embeddings = node2vec.get_embedding() + (seed * 6)
            embeddings_lst.append(pd.DataFrame(data=embeddings, columns=['x', 'y']))

        plt.figure(len(gs) + i)
        plt.scatter(x=embeddings_lst[0]['x'], y=embeddings_lst[0]['y'], c='g', label=f'seed={0}')
        for index, row in embeddings_lst[0].iterrows():
            plt.annotate(index, (row['x'], row['y']))
        plt.scatter(x=embeddings_lst[1]['x'], y=embeddings_lst[1]['y'], c='b', label=f'seed={1}')
        for index, row in embeddings_lst[1].iterrows():
            plt.annotate(index, (row['x'], row['y']))
        plt.scatter(x=embeddings_lst[2]['x'], y=embeddings_lst[2]['y'], c='r', label=f'seed={2}')
        for index, row in embeddings_lst[2].iterrows():
            plt.annotate(index, (row['x'], row['y']))
        plt.scatter(x=embeddings_lst[3]['x'], y=embeddings_lst[3]['y'], c='y', label=f'seed={3}')
        for index, row in embeddings_lst[3].iterrows():
            plt.annotate(index, (row['x'], row['y']))

        # plt.scatter(x=embeddings_lst[4]['x'], y=embeddings_lst[4]['y'], c='k', label=4)


        plt.title('Graph Embeddings With Diffrent Seeds')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.legend()
        plt.show()
        a = 1

class Drawer:

    def __init__(self, max_err):
        self.orig_df = pd.read_csv(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\Combined_Results\statistics.csv')
        df = self.orig_df[self.orig_df[StatisticsParams.error] != '-']
        df[StatisticsParams.error] = df[StatisticsParams.error].astype(float)
        df[StatisticsParams.runtime] = df[StatisticsParams.runtime].apply(lambda rtime: rtime.split('.')[0])
        df[StatisticsParams.runtime] = df[StatisticsParams.runtime].apply(
            lambda rtime: "00:" + rtime if len(rtime) == 5 else rtime)
        df[StatisticsParams.runtime] = df[StatisticsParams.runtime].apply(lambda rtime: (
                datetime.datetime.strptime(rtime.split('.')[0], "%H:%M:%S") - datetime.datetime(1900, 1,
                                                                                                1)).total_seconds())
        self.df_converge = df[df['Error'] < max_err]

    def draw_rtime_as_func_of_num_node(self):
        plt = px.scatter(self.df_converge,
                         color=StatisticsParams.centrality,
                         x=StatisticsParams.num_nodes,
                         y=StatisticsParams.runtime,
                         size=StatisticsParams.num_edges,
                         size_max=30,
                         facet_col=StatisticsParams.centrality,
                         height=500)
        plt.show()

    def draw_rtime_as_func_of_num_edges(self):
        plt = px.scatter(self.df_converge,
                         color=StatisticsParams.centrality,
                         x=StatisticsParams.num_edges,
                         y=StatisticsParams.runtime,
                         size=StatisticsParams.num_nodes,
                         size_max=20,
                         facet_col=StatisticsParams.centrality,
                         height=500)

        plt.show()


if __name__ == '__main__':
    # drawer = Drawer(1e-05)
    # drawer.draw_rtime_as_func_of_num_node()
    # drawer.draw_rtime_as_func_of_num_edges()
    draw_embeddings()
