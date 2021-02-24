import datetime
from Utils.CommonStr import HyperParams
from Tests.Tools import saver
from Components.RBC_ML.RbcML import learn_models
from Components.RBC_REG.RBC import RBC
from Utils.CommonStr import LearningParams
from Utils.CommonStr import Centralities
from Tests.RBC_ML.GraphGenerator import GraphGenerator
from Tests.RBC_ML.ParamsManager import ParamsManager


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
        graphs = self.graphs_generator.generate_by_centrality()
        for i in range(1, len(graphs)):
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

        start_time = datetime.datetime.now()
        try:
            t_model, r_model, final_error = learn_models(learning_params, nodes_mapping_reverse)
            runtime = datetime.datetime.now() - start_time
            rbc_pred = rbc_handler.compute_rbc(g, r_model, t_model)
            print(f'\n\ntest of figure_{test_num}, RBC Prediction returned - {rbc_pred}')
            params_manager.save_params_statistics(t_model, r_model, final_error, runtime, rbc_pred)
        except Exception as e:
            saver.save_info_stuck(self.centrality, adj_matrix, learning_target, learning_params, str(e))


if __name__ == '__main__':
    spbc_tester = CentralityTester(Centralities.Closeness)
    degree_tester = CentralityTester(Centralities.Degree)
    eigenvector_tester = CentralityTester(Centralities.Eigenvector)
    closeness_tester = CentralityTester(Centralities.Closeness)
    testers = [spbc_tester, degree_tester, eigenvector_tester, closeness_tester]

    for tester in testers:
        tester.test_centrality()
