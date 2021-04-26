import os

from Utils.CommonStr import RoutingTypes


class Single_Graph_Fixed_Routing_SPBC_4_nodes:

    def __init__(self):
        self.n_routing_per_graph = 1
        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\4_nodes_fixed_rbc\SPBC'
        self.train_graphs_path = f'{self.root_path}\\train'
        self.validation_graphs_path = f'{self.root_path}\\validation'
        self.test_graphs_path = f'{self.root_path}\\test'
        self.description = f'Single Graph, , 4 Nodes, Fixed Routing'


class Single_Graph_Similar_Routing_SPBC_7_nodes:
    def __init__(self):
        self.n_routing_per_graph = 8
        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\7_nodes_similar_rbc\SPBC'
        self.train_graphs_path = f'{self.root_path}\\train'
        self.validation_graphs_path = f'{self.root_path}\\validation'
        self.test_graphs_path = f'{self.root_path}\\test'
        self.description = f'Single Graph, , 7 Nodes, Similar Routing'


class Single_Graph_Fixed_Routing_SPBC_9_nodes:
    def __init__(self):
        self.n_routing_per_graph = 1
        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\9_nodes_fixed_rbc\SPBC'
        self.train_graphs_path = f'{self.root_path}\\train'
        self.validation_graphs_path = f'{self.root_path}\\validation'
        self.test_graphs_path = f'{self.root_path}\\test'
        self.description = f'Single Graph, , 9 Nodes, Fixed Routing'


class Single_Graph_Fixed_Routing_SPBC_11_nodes:
    def __init__(self):
        self.n_routing_per_graph = 1
        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_1\Data\11_nodes_fixed_rbc\SPBC'
        self.train_graphs_path = f'{self.root_path}\\train'
        self.validation_graphs_path = f'{self.root_path}\\validation'
        self.test_graphs_path = f'{self.root_path}\\test'
        self.description = f'Single Graph, , 11 Nodes, Fixed Routing'


class DifferentGraphs_9Nodes_15Edges:
    def __init__(self):
        self.n_routing_per_graph = 1
        self.root_path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Experiments\Experiments_2\Data\SPBC\graphs_1'
        self.train_graphs_path = f'{self.root_path}\\train'
        self.validation_graphs_path = f'{self.root_path}\\validation'
        self.test_graphs_path = f'{self.root_path}\\test'
        self.description = r'Different Graphs, 9 Nodes, 15 Edges'
