import numpy as np
from objprint import objprint

from scipy.spatial.distance import cdist
from typing import Callable, Any
from pyfmto.framework import Server, SyncDataManager
from pyfmto.utilities import logger

from .fdemd_utils import (
    RadialBasisFunctionNetwork as RBFNetwork,
    Actions, ClientPackage, init_samples
)


class FdemdServer(Server):
    """
    ensemble_size: 20
    epoch: 5         # local epoch
    optimizer: sgd   # optimizer of RBF network, sgd/m-sgd/max-gd
    lr: 0.06         # learning rate
    alpha: 1.0       # noisy
    """

    def __init__(self, **kwargs):
        super().__init__()
        # centers, spreads, w and b can be broadcast to clients
        kwargs = self.update_kwargs(kwargs)
        self.ensemble_size = kwargs['ensemble_size']
        self.model_args = {
            'epoch': kwargs['epoch'],
            'optimizer': kwargs['optimizer'],
            'lr': kwargs['lr'],
            'alpha': kwargs['alpha'],
        }

        # initialize in self._router_pull_init
        self.dim = None
        self.obj = None
        self.lb = None
        self.ub = None
        self.model = None
        self.d_aux_size = None
        self.kernel_size = None

        self.d_aux = None
        self.clients_data = SyncDataManager()

    def handle_request(self, client_data: ClientPackage):
        action_map: dict[Actions, Callable[[ClientPackage], Any]] = {
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
            raise ValueError(f"Unknown requested action: {client_data.action}")

    def _push_init(self, pkg: ClientPackage):
        # for the same init logic with original implementation, init using cid==1 data
        if pkg.cid == 1 and self.model is None:
            self.dim = pkg.dim
            self.obj = pkg.obj
            kernel_size = 2*self.dim + 1
            weight = np.random.randn(kernel_size, self.obj)
            bias = np.random.randn(1, self.obj)
            self.model = RBFNetwork(dim=self.dim, obj=self.obj, kernel_size=kernel_size, **self.model_args)
            self.model.sync_manuel(weight=weight, bias=bias)
            self.d_aux_size = pkg.init_size
            logger.info(f"D aux size is {self.d_aux_size}")
            self.kernel_size = kernel_size
            self.lb = pkg.lb
            self.ub = pkg.ub
            logger.debug(f'Client package is \n{objprint(pkg)}')
        return 'init success'

    def _pull_init(self, pkg: ClientPackage):
        return None if self.model is None else self.model.params

    def _push_update(self, pkg: ClientPackage):
        self.clients_data.update_src(pkg.cid, pkg.version, pkg)
        return 'success'

    def _pull_update(self, pkg: ClientPackage):
        return self.clients_data.get_res(0, pkg.version)

    def aggregate(self):
        ver = self.clients_data.available_src_ver
        if self.should_agg:
            logger.debug(f"Aggregating version {ver}")
            self.d_aux = init_samples(self.dim, self.lb, self.ub, self.d_aux_size)
            self._distill(ver)
            self.clients_data.update_res(0, version=ver, data=self.model.params)

    @property
    def should_agg(self):
        n_src = self.clients_data.num_clients
        n_clt = self.num_clients
        src_ver = self.clients_data.available_src_ver
        res_ver = self.clients_data.lts_res_ver(0)
        a = src_ver > res_ver
        b = n_src == n_clt
        c = self.d_aux_size is not None
        return a and b and c

    def _distill(self, version):
        all_centers = []
        all_weights = []
        all_biases = []
        all_std = []
        for cid in self.sorted_ids:
            pkg: ClientPackage = self.clients_data.get_src(cid, version)
            all_centers.append(pkg.network.get('_centers'))
            all_weights.append(pkg.network.get('_weight'))
            all_biases.append(pkg.network.get('_bias'))
            all_std.append(pkg.network.get('_std'))
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

        variance_centers = sum_squared_diff_centers / self.num_clients
        variance_weights = sum_squared_diff_weights / self.num_clients
        variance_biases = sum_squared_diff_biases / self.num_clients
        variance_std = sum_squared_diff_std / self.num_clients

        f_sudo = self._ensemble_predict(variance_centers, variance_std, variance_weights, variance_biases,
                                        mean_centers, mean_weights, mean_biases, mean_std)
        self.model.train(self.d_aux, f_sudo)

    def _ensemble_predict(self, variance_centers, variance_std, variance_weights, variance_biases,
                          mean_centers, mean_weights, mean_biases, mean_std):
        predictions = []
        num_test_samples = self.d_aux.shape[0]
        dim = self.d_aux.shape[1]

        for _ in range(self.ensemble_size):
            mean_centers = np.array([mean_centers]) if not isinstance(mean_centers, np.ndarray) else mean_centers
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