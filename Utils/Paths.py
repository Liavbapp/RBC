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
        paths = [
                    r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\4_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.num_seeds_per_routing
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
        paths = [
                    r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\9_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.n_seeds_per_routing
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
        paths = [
                    r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\11_nodes_fixed_rbc\Raw_Data\SPBC\0'] * self.n_seeds_per_routing
        return paths

    def get_description(self):
        return f'Single Graph {self.total_seeds_per_graph} seeds, Fixed Routing, 11 Nodes'


class DifferentGraphs9Nodes15Edges:
    def __init__(self):
        self.n_routing_per_graph = 1

        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_2\Graphs\SPBC'
        self.train_graphs = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_2\Graphs\SPBC\train'
        self.validation_graphs = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_2\Graphs\SPBC\validation'
        self.test_graphs = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_2\Graphs\SPBC\test'

        self.description = r'Diffrent Graph with 9 Nodes, 15 Edges'
