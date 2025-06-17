import numpy as np
import yaml
from collections import defaultdict
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Callable
from pyfmto.framework import Server, ClientPackage, ServerPackage, Actions
from pyfmto.utilities import logger

from pyfmto.algorithms.TS import init_samples
from .fdemd_utils import RadialBasisFunctionNetwork as RBFNetwork, AggData


class FdemdServer(Server):

    def __init__(self):
        super().__init__()
        with open(Path(__file__).parent / 'fdemd.yaml') as f:
            params = yaml.safe_load(f)
        # centers, spreads, w and b can be broadcast to clients
        server_params = params.get('server', {})
        self.model_args = params.get('model', {})
        self.ensemble_size = server_params.get('ensemble_size', 20)

        # initialize in self._router_pull_init
        self.dim = None
        self.obj = None
        self.x_lb = None
        self.x_ub = None
        self.model = None
        self.d_aux_size = None
        self.kernel_size = None

        self.d_aux = None
        self.clients_data = defaultdict(list)
        self.agg_res = []

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        action_map: dict[Actions, Callable[[ClientPackage], ServerPackage]] = {
            Actions.PUSH_INIT: self._push_init,
            Actions.PULL_INIT: self._pull_init,
            Actions.PUSH_UPDATE: self._push_update,
            Actions.PULL_UPDATE: self._pull_update,
        }

        action = action_map.get(client_data.action)
        if action is not None:
            return action(client_data)
        else:
            logger.error(f"Unknown client requested action: {client_data.action}")
            return ServerPackage('status', {'status': 'error'})

    def _push_init(self, client_data: ClientPackage) -> ServerPackage:
        # for the same init logic with original implementation, init using cid==1 data
        if client_data.cid==1 and self.model is None:
            self.dim = client_data.data['dim']
            self.obj = client_data.data['obj']
            kernel_size = 2*self.dim + 1
            weight = np.random.randn(kernel_size, self.obj)
            bias = np.random.randn(1, self.obj)
            self.model = RBFNetwork(dim=self.dim, obj=self.obj, kernel_size=kernel_size, **self.model_args)
            self.model.sync_manuel(weight=weight, bias=bias)
            self.d_aux_size = client_data.data['init_size']
            self.kernel_size = kernel_size
            self.x_lb = client_data.data['lb']
            self.x_ub = client_data.data['ub']
        return ServerPackage('status', {'status': 'success'})

    def _pull_init(self, client_data: ClientPackage) -> ServerPackage:
        if self.model is None:
            return ServerPackage('init', None)
        else:
            return ServerPackage('init', self.model.params)

    def _push_update(self, client_data: ClientPackage) -> ServerPackage:
        cid = client_data.cid
        self.clients_data[cid].append(client_data.data)
        return ServerPackage('success', {'status': 'success'})

    def _pull_update(self, client_data: ClientPackage) -> ServerPackage:
        if self.agg_res:
            return ServerPackage('update', self.agg_res[-1])
        return ServerPackage('update', None)

    def aggregate(self, client_id):
        curr_ver = len(self.agg_res)
        ids = self.sorted_ids
        vers = np.asarray([len(self.clients_data[cid]) for cid in ids])
        if np.all(vers > curr_ver):
            self._distill(ids, curr_ver)
            self.agg_res.append(AggData(version=curr_ver+1, src_num=len(ids), agg_res=self.model.params))

    def _distill(self, client_ids, src_index):
        all_centers = []
        all_weights = []
        all_biases = []
        all_std = []

        for cid in client_ids:
            all_centers.append(self.clients_data[cid][src_index].get('_centers'))
            all_weights.append(self.clients_data[cid][src_index].get('_weight'))
            all_biases.append(self.clients_data[cid][src_index].get('_bias'))
            all_std.append(self.clients_data[cid][src_index].get('_std'))
        all_centers = np.asarray(all_centers)
        all_weights = np.asarray(all_weights)
        all_biases = np.asarray(all_biases)
        all_std = np.asarray(all_std)

        mean_centers = np.mean(all_centers, axis=0)
        mean_weights = np.mean(all_weights, axis=0)
        mean_biases = np.mean(all_biases, axis=0)
        mean_std = np.mean(all_std, axis=0)

        sum_squared_diff_centers = np.sum(np.square(all_centers - mean_centers), axis=0)
        sum_squared_diff_weights = np.sum(np.square(all_weights - mean_weights), axis=0)
        sum_squared_diff_biases = np.sum(np.square(all_biases - mean_biases), axis=0)
        sum_squared_diff_std = np.sum(np.square(all_std - mean_std), axis=0)

        num_clients = len(client_ids)
        variance_centers = sum_squared_diff_centers / num_clients
        variance_weights = sum_squared_diff_weights / num_clients
        variance_biases = sum_squared_diff_biases / num_clients
        variance_std = sum_squared_diff_std / num_clients

        self.d_aux = init_samples(self.x_lb, self.x_ub, self.dim, self.d_aux_size)
        f_sudo = self._ensemble_predict(variance_centers, variance_std, variance_weights, variance_biases,
                                        mean_centers, mean_weights, mean_biases, mean_std)
        self.model.train(self.d_aux, f_sudo)

    def _ensemble_predict(self, variance_centers, variance_std, variance_weights, variance_biases,
                          mean_centers, mean_weights, mean_biases, mean_std):
        predictions = []
        num_test_samples = self.d_aux.shape[0]
        dim = self.d_aux.shape[1]

        for _ in range(self.ensemble_size):
            sampled_centers = [
                np.random.multivariate_normal(mean_centers[j], np.diag(variance_centers[j]), 1 ) for j in range(self.kernel_size)
            ]

            sampled_weights = [
                np.random.normal(loc=mean_weights[j], scale=variance_weights[j], size=1) for j in range(self.kernel_size)
            ]

            sampled_spreads = [
                np.random.normal(loc=mean_std[j], scale=variance_std[j], size=1) for j in range(self.kernel_size)
            ]

            sampled_bias = np.random.normal(loc=mean_biases[0], scale=variance_biases[0], size=1)

            centers_array = np.array(sampled_centers).reshape(self.kernel_size, dim)
            center_to_test_distances = self._dist(centers_array, self.d_aux.T)

            spread_matrix = np.tile( np.array(sampled_spreads).reshape(-1, 1), (1, num_test_samples))
            hidden_output = np.exp(-(center_to_test_distances / spread_matrix) ** 2).T

            prediction = np.dot(hidden_output, sampled_weights) + sampled_bias
            predictions.append(prediction)
        predictions = np.asarray(predictions)
        return np.mean(predictions, axis=0)

    @staticmethod
    def _dist(mat1, mat2):
        """
        rewrite euclidean distance function in Matlab: dist
        :param mat1: matrix 1, M x N
        :param mat2: matrix 2, N x R
        output: Mat3. M x R
        """
        mat2 = mat2.T
        return cdist(mat1, mat2)