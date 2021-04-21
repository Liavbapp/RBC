import os

from Utils.CommonStr import RoutingTypes


class Single_Graph_Fixed_Routing_SPBC_4_nodes:
    def __init__(self, num_seeds_routing):
        self.num_seeds_per_routing = num_seeds_routing
        self.n_routing_per_graph = 1
        self.total_seeds_per_graph = self.num_seeds_per_routing * self.n_routing_per_graph
        self.n_graphs = 1
        self.n_seeds_graph = num_seeds_routing
        self.routing_type = RoutingTypes.fixed

    def get_paths(self):
        paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\4_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.num_seeds_per_routing
        return paths

    def get_description(self):
        return f'Single Graph {self.total_seeds_per_graph} seeds, Fixed Routing, 4 Nodes'


class Single_Graph_Similar_Routing_SPBC_7_nodes:
    def __init__(self, num_seeds_routing):
        self.n_seeds_per_routing = num_seeds_routing
        self.n_graphs = 1
        self.n_routing_per_graph = 10
        self.total_seeds_per_graph = self.n_seeds_per_routing * self.n_routing_per_graph
        self.num_seeds_graph_routing = num_seeds_routing
        self.n_seeds_graph = num_seeds_routing
        self.routing_type = RoutingTypes.similar

    def get_paths(self):
        paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Experiments\\Experiments_1\\Data' \
                  f'\\7_nodes_similar_rbc\\Raw_Data\\SPBC\\{str(i)}' for i in range(0, 11)] * self.n_seeds_per_routing
        return paths

    def get_description(self):
        return f'Single Graph {self.total_seeds_per_graph} seeds, Similar Routing, 7 Nodes'


class Single_Graph_Fixed_Routing_SPBC_9_nodes:
    def __init__(self, num_seeds_routing):
        self.n_seeds_per_routing = num_seeds_routing
        self.n_graphs = 1
        self.n_routing_per_graph = 1
        self.total_seeds_per_graph = self.n_seeds_per_routing * self.n_routing_per_graph
        self.num_seeds_graph_routing = num_seeds_routing
        self.n_seeds_graph = num_seeds_routing
        self.routing_type = RoutingTypes.similar

    def get_paths(self):
        paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\9_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.n_seeds_per_routing
        return paths

    def get_description(self):
        return f'Single Graph {self.total_seeds_per_graph} seeds, Fixed Routing, 9 Nodes'


class Single_Graph_Fixed_Routing_SPBC_11_nodes:
    def __init__(self, num_seeds_routing):
        self.n_seeds_per_routing = num_seeds_routing
        self.n_graphs = 1
        self.n_routing_per_graph = 1
        self.total_seeds_per_graph = self.n_seeds_per_routing * self.n_routing_per_graph
        self.num_seeds_graph_routing = num_seeds_routing
        self.n_seeds_graph = num_seeds_routing
        self.routing_type = RoutingTypes.similar

    def get_paths(self):
        paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\11_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.n_seeds_per_routing
        return paths

    def get_description(self):
        return f'Single Graph {self.total_seeds_per_graph} seeds, Fixed Routing, 11 Nodes'


# train_combined = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\15',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\9',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\10',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\12',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\396',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\397',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\398',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\399',
#                   r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\400'
#                   ]


#
# class Same_Graph_Same_Routing_15_nodes:
#     paths = [r'C:\Users\LiavB\PycharmProjects\RBC\results\matrices\SPBC\15_nodes\2_edges\9'] * 50
#     graphs_desc = f'Same Graph {len(paths)} times, With Same Routing, 15 Nodes'
#


#
# class Same_Graph_Different_Routing_4_nodes:
#     paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\SPBC\\4_nodes\\2_edges\\{str(i)}'
#              for i in range(390, 401)]
#     graphs_desc = f'Same Graph {len(paths)} times, Different Routing, 4 Nodes'
#
#

#
# class Load_Diff_13_nodes:
#     # paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\54_edges\\0']
#     pi = 1
#
#
#     paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\39_edges\\{str(i)}' for i in range(0, 3)]
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\40_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\41_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\42_edges\\{str(i)}' for i in range(0, 4)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\43_edges\\{str(i)}' for i in range(0, 6)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\44_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\45_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\46_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\48_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\49_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\50_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\51_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\53_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\54_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\55_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\56_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\57_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\58_edges\\{str(i)}' for i in range(0, 5)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\59_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\60_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\61_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\62_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\63_edges\\{str(i)}' for i in range(0, 6)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\64_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\65_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\66_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\67_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\68_edges\\{str(i)}' for i in range(0, 5)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\69_edges\\{str(i)}' for i in range(0, 1)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\70_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\71_edges\\{str(i)}' for i in range(0, 4)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\72_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\73_edges\\{str(i)}' for i in range(0, 3)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\74_edges\\{str(i)}' for i in range(0, 2)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\75_edges\\{str(i)}' for i in range(0, 4)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\76_edges\\{str(i)}' for i in range(0, 4)] * pi
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\13_nodes\\77_edges\\{str(i)}' for i in range(0, 4)] * pi
#     graphs_desc = f'kawbanga22'
#
#
# class LoadCentrality_Diffrent_Routing_5nodes:
#     pi = 1
#
#     paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\5_edges\\{str(i)}'
#              for i in range(1, 4)]
#     # paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\5_edges\\{str(i)}'
#     #          for i in range(0, 1)]
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\5_edges\\{str(i)}'
#              for i in range(5, 7)] * pi
#     # paths += ['C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\4_edges\\0']
#     #
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\6_edges\\{str(i)}'
#              for i in range(0, 1)] * pi
#     #
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\7_edges\\{str(i)}'
#               for i in range(0, 5)] * pi
#     #
#     paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\10_edges\\{str(i)}'
#               for i in range(0, 1)] * pi
#     #
#     # paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\2_edges\\{str(i)}'
#     #           for i in range(0, 1)] * pi
#     #
#     # paths += [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\Load\\5_nodes\\3_edges\\{str(i)}'
#     #           for i in range(0, 1)] * pi
#     graphs_desc = f'kawbanga'
#
#
# class ManyGraphs:
#     paths = [x[0] for x in os.walk(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\Load') if x[0][-1] == '0'][:18]
#     paths[0] = paths[4]
#     del paths[4]
#     graphs_desc = "many_graphs"
