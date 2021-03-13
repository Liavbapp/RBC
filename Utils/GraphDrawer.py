import plotly.express as px
import pandas as pd
import datetime

from Utils.CommonStr import StatisticsParams


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
    drawer = Drawer(1e-05)
    # drawer.draw_rtime_as_func_of_num_node()
    drawer.draw_rtime_as_func_of_num_edges()