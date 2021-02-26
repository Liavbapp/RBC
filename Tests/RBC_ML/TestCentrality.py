import datetime
import json
import sys
sys.path.append('C:\\Users\\LiavB\\PycharmProjects\\RBC')
from Components.RBC_ML.Optimizer import Optimizer
from Components.RBC_ML.RbcNetwork import RbcNetwork
from Utils.CommonStr import HyperParams
from Tests.Tools import saver
from Components.RBC_ML.RbcML import learn_models
from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import LearningParams
from Utils.CommonStr import Centralities
from Tests.RBC_ML.GraphGenerator import GraphGenerator
from Tests.RBC_ML.ParamsManager import ParamsManager

import sys
sys.path.append('C:\\Users\\LiavB\\PycharmProjects\\RBC\\Components')


def init_model(learning_params):
    num_nodes = len(learning_params[LearningParams.adjacency_matrix][0])
    use_sigmoid = learning_params[LearningParams.sigmoid]
    pi_max_err = learning_params[LearningParams.hyper_parameters][HyperParams.pi_max_err]
    eigenvector_method = learning_params[LearningParams.eigenvector_method]
    consider_traffic_paths = learning_params[LearningParams.consider_traffic_paths]
    device = learning_params[LearningParams.device]
    dtype = learning_params[LearningParams.dtype]
    model = RbcNetwork(num_nodes=num_nodes, use_sigmoid=use_sigmoid, pi_max_err=pi_max_err,
                       eigenvector_method=eigenvector_method, device=device, dtype=dtype,
                       cosnider_traffic_paths=consider_traffic_paths)
    return model


def init_optimizer(model, hyper_params):
    optimizer_name = hyper_params[HyperParams.optimizer]
    learning_rate = hyper_params[HyperParams.learning_rate]
    optimizer = Optimizer(model, optimizer_name, learning_rate)
    return optimizer


def get_rbc_handler(learning_params, hyper_params):
    eigenvector_method = learning_params[LearningParams.eigenvector_method]
    pi_max_error = hyper_params[HyperParams.pi_max_err]
    device = learning_params[LearningParams.device]
    dtype = learning_params[LearningParams.dtype]
    rbc_handler = RBC(eigenvector_method, pi_max_error, device, dtype)

    return rbc_handler


class CentralityTester():

    def __init__(self, centrality=Centralities.SPBC):
        self.centrality = centrality
        self.graphs_generator = GraphGenerator(centrality)

    def test_centrality(self):
        graphs = self.graphs_generator.custom_graph()
        for i in range(0, len(graphs)):
            self.test_centrality_on_graph(graphs[i], i)

    def test_centrality_on_graph(self, g, test_num):
        print(f' Testing {self.centrality} Centrality - test number {test_num}')
        params_manager = ParamsManager(g, self.centrality)
        hyper_params = params_manager.hyper_params
        learning_params = params_manager.learning_params
        rbc_handler = get_rbc_handler(learning_params, hyper_params)
        adj_matrix = learning_params[LearningParams.adjacency_matrix]
        learning_target = learning_params[LearningParams.target]
        nodes_mapping_reverse = {k: v for k, v in enumerate(list(g.nodes()))}
        model = init_model(learning_params)
        optimizer = init_optimizer(model, hyper_params)
        optimizer_params = json.dumps(optimizer.get_optimizer_params())

        start_time = datetime.datetime.now()

        try:
            t_model, r_model, final_error = learn_models(model, g, learning_params, nodes_mapping_reverse, optimizer)
            runtime = datetime.datetime.now() - start_time
            rbc_pred = rbc_handler.compute_rbc(g, r_model, t_model)
            print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
            params_manager.save_params_statistics(t_model, r_model, final_error, runtime, rbc_pred, optimizer_params)
        except Exception as e:
            print(str(e))
            saver.save_info_stuck(self.centrality, adj_matrix, learning_target, learning_params, str(e), optimizer_params)


if __name__ == '__main__':
    spbc_tester = CentralityTester(Centralities.SPBC)
    degree_tester = CentralityTester(Centralities.Degree)
    eigenvector_tester = CentralityTester(Centralities.Eigenvector)
    closeness_tester = CentralityTester(Centralities.Closeness)
    testers = [spbc_tester, degree_tester, eigenvector_tester, closeness_tester]


    for tester in testers:
        tester.test_centrality()
