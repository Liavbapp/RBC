import concurrent
import datetime
import json
import sys
from concurrent.futures.thread import ThreadPoolExecutor

sys.path.append('C:\\Users\\LiavB\\PycharmProjects\\RBC')
from Components.RBC_ML.Optimizer import Optimizer
from Components.RBC_ML.RbcNetwork import RbcNetwork
from Utils.CommonStr import HyperParams
from Components.RBC_ML.RbcML import learn_models
from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import LearningParams
from Utils.CommonStr import Centralities
from Utils.GraphGenerator import GraphGenerator
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
        graphs = self.graphs_generator.generate_rand_graphs()
        results = []
        for i in range(0, len(graphs)):
            results.append(self.load_params_statistics(graphs[i], i))
        return results

    def load_params_statistics(self, g, test_num):
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
        print(f'start time: {start_time}')

        try:
            t_model, r_model, final_error = learn_models(model, g, learning_params, nodes_mapping_reverse, optimizer)
            runtime = datetime.datetime.now() - start_time
            rbc_pred = rbc_handler.compute_rbc(g, r_model, t_model)
            print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
            params_manager.prepare_params_statistics(t_model.detach(), r_model.detach(), final_error, runtime, rbc_pred.detach(), optimizer_params)
        except Exception as e:
            print(str(e))
            params_manager.prepare_stuck_params_statistics(self.centrality, adj_matrix.detach(), learning_target, learning_params, str(e), optimizer_params)

        return params_manager


if __name__ == '__main__':
    spbc_tester = CentralityTester(Centralities.SPBC)
    degree_tester = CentralityTester(Centralities.Degree)
    eigenvector_tester = CentralityTester(Centralities.Eigenvector)
    closeness_tester = CentralityTester(Centralities.Closeness)
    testers = [spbc_tester, degree_tester, eigenvector_tester, closeness_tester]

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(testers)) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(CentralityTester.test_centrality, tester): tester for tester in testers}
        for future in concurrent.futures.as_completed(future_to_url):
            tester_identity = future_to_url[future].centrality
            try:
                params_managers = future.result()
                [param_manager.save_params_statistics() for param_manager in params_managers]
            except Exception as exc:
                print('%r generated an exception: %s' % (tester_identity, exc))

