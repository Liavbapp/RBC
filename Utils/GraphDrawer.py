import networkx as nx
import plotly.express as px
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from Utils.CommonStr import StatisticsParams
from Utils.GraphGenerator import GraphGenerator
from karateclub import Diff2Vec, RandNE, GLEE, NetMF, NNSED, DANMF, MNMF, BigClam, SymmNMF, SocioDim, NodeSketch, \
    BoostNE, Walklets, GraRep, NMFADMM, LaplacianEigenmaps, FeatherNode, AE, DeepWalk, GraphWave, MUSAE, Node2Vec, Role2Vec, GL2Vec


# diff2vec_sizes = [(seed * 5), 50, diffusion_cover=30, diffusion_number=20]
    # LaplacianEigenmaps  = [(seed * 0.5), 3.6]
    # glee_sizes = [(seed * 1.5), 12]
    # nnsed_size = [(seed * 0.01), 0.08]
    # RANDNE = [(seed * 7), 40]
    # mnmf = [(seed * 5), 45]
    # bigclam = [(seed * 10), 150]
    # SymmNMF
    # SocioDim = [(seed * 1), 7]
    # Node2Sketch = [(seed * 301), 2500]
    # NETMF = [(seed *10), 100]
    # BoostNE -> not sucessfully
    #Walklet -> not sucessfully
    #GraRep -> not sucess
    # NMFADMM -> [(seed * 1.5) , 12]
def plot_orig_graph(g, i):
    plt.figure(0)
    nx.draw(g, with_labels=True)


def draw_embeddings():

    gs = GraphGenerator('SPBC').graphs_for_embeddings_show()
    WITH_ORIG = True
    node2vec = LaplacianEigenmaps(dimensions=2)
    num_seeds = 7
    color_map = {0: 'g', 1: 'b', 2: 'r', 3: 'y', 4: 'k', 5: 'c', 6: 'm'}
    # graph_wave = GraphWave(dimensions=2)
    for i in range(0, len(gs)):
        plot_orig_graph(gs[i], i)
        embeddings_lst_norm = []
        embedding_lst = []
        for seed in range(num_seeds):
            node2vec.seed = seed
            node2vec.fit(gs[i])
            actual_embeddings = node2vec.get_embedding()
            embeddings_norm = actual_embeddings + (seed * 0.5)
            embedding = actual_embeddings + 3.6
            embeddings_lst_norm.append(pd.DataFrame(data=embeddings_norm, columns=['x', 'y']))
            embedding_lst.append(pd.DataFrame(data=embedding, columns=['x', 'y']))

        plt.figure(len(gs) + i)

        for j in range(num_seeds):
            plt.scatter(x=embeddings_lst_norm[j]['x'], y=embeddings_lst_norm[j]['y'], c=color_map[j], label=f'norm_seed={j}')
            for index, row in embeddings_lst_norm[j].iterrows():
                plt.annotate(index, (row['x'], row['y']))
            if WITH_ORIG:
                plt.scatter(x=embedding_lst[j]['x'], y=embedding_lst[j]['y'], c=color_map[j], label=f'actual_seed={j}')
                for index, row in embedding_lst[j].iterrows():
                    plt.annotate(index, (row['x'], row['y']))

        plt.title('Graph Embeddings With Diffrent Seeds')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.legend()
        plt.show()


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
