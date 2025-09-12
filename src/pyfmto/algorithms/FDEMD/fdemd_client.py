import numpy as np
from numpy import ndarray
from pyfmto.framework import Client, record_runtime

from .fdemd_utils import (
    RadialBasisFunctionNetwork as RBFNetwork,
    GeneticAlgorithm, Actions, ClientPackage)
from ...utilities import logger

ga_op = GeneticAlgorithm()


class FdemdClient(Client):
    """
    lg_type: LG
    max_gen: 20
    epoch: 5         # local epoch
    optimizer: sgd   # optimizer of RBF network, sgd/m-sgd/max-gd
    lr: 0.06         # learning rate
    alpha: 1.0       # noisy
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        kwargs = self.update_kwargs(kwargs)

        self.lg_type = kwargs['lg_type']
        self.max_gen = kwargs['max_gen']
        model_args = {
            'epoch': kwargs['epoch'],
            'optimizer': kwargs['optimizer'],
            'lr': kwargs['lr'],
            'alpha': kwargs['alpha'],
        }
        self.client_model = RBFNetwork(dim=self.dim, obj=self.obj, kernel_size=2 * self.dim + 1, **model_args)
        self.server_model = RBFNetwork(dim=self.dim, obj=self.obj, kernel_size=2 * self.dim + 1)

        self.version = 1

    def optimize(self):
        if not self.initialized:
            self._initializing()
        else:
            self._optimizing()

    def _initializing(self):
        logger.debug(f"Client {self.id} initializing")
        self.push_init_data()
        self.pull_init_model()

    def push_init_data(self):
        init_data = {'init_size': self.solutions.fe_init,
                     'dim': self.dim, 'obj': self.obj,
                     'lb': self.lb, 'ub': self.ub}
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_INIT, **init_data)
        self.request_server(pkg)

    def pull_init_model(self):
        # this pull only for the first time
        # and data only for client initialize model
        # so centers and std are not needed
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_INIT)
        res = self.request_server(pkg, repeat=100)
        self.client_model.sync_auto(res)

    @record_runtime('Total')
    def _optimizing(self):
        logger.debug(f"Client {self.id} optimizing")
        self.client_model.train(self.x, self.y)
        self.push_client_model()
        self.pull_server_model()
        next_x = self._find_next_x()
        next_y = self.problem.evaluate(next_x)
        self.solutions.append(next_x.reshape(-1, self.dim), next_y.reshape(-1, self.obj))
        self.client_model.sync_auto(self.server_model.params)
        self.record_round_info('FE', self.solutions.size)
        self.record_round_info('Best', f"{self.solutions.y_min:.2f}")
        self.version += 1

    @record_runtime('Push')
    def push_client_model(self):
        pkg = ClientPackage(
            cid=self.id,
            action=Actions.PUSH_UPDATE,
            version=self.version,
            network=self.client_model.params
        )
        self.request_server(pkg)

    @record_runtime('Pull')
    def pull_server_model(self):
        # this pull operation is for the optimization rounds
        # the pulled model is for prediction and reinitialize the local model
        # the prediction is based on all 4 params of the global model
        pkg = ClientPackage(cid=self.id, version=self.version, action=Actions.PULL_UPDATE)
        res = self.request_server(pkg, repeat=100)
        self.server_model.sync_auto(res)

    @record_runtime('Find')
    def _find_next_x(self):
        lb = self.lb if isinstance(self.lb, ndarray) else self.lb * np.ones(self.dim)
        ub = self.ub if isinstance(self.ub, ndarray) else self.ub * np.ones(self.dim)
        _, chosen_pop, pop, pop_uncertainty = ga_op.RCGA(obj_func=self.lcb, lb=lb, ub=ub, max_iter=self.max_gen,
                                                         particle_output=True)
        return chosen_pop.flatten()

    def lcb(self, new_pops):

        num_pop = new_pops.shape[0]
        f_g_hat = self.server_model.predict(new_pops).reshape(num_pop, 1)
        f_l_hat = self.client_model.predict(new_pops).reshape(num_pop, 1)

        if self.lg_type == 'L':
            f_mean_hat = np.mean(f_l_hat, axis=1).reshape(num_pop, 1)
            tmp = (f_l_hat - f_mean_hat) ** 2
            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (f_l_hat.shape[1] - 1)
        elif self.lg_type == 'G':
            f_mean_hat = f_g_hat
            tmp = (f_l_hat - f_mean_hat) ** 2
            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (f_l_hat.shape[1] - 1)
        elif self.lg_type == 'LG':
            local_mean = np.mean(f_l_hat, axis=1).reshape(num_pop, 1)
            combined_sum = np.hstack((local_mean, f_g_hat))
            f_mean_hat = np.mean(combined_sum, axis=1).reshape(-1, 1)
            combined_f_hat = np.hstack((f_l_hat, f_g_hat))
            tmp = (combined_f_hat - f_mean_hat) ** 2
            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (combined_f_hat.shape[1] - 1)
        else:
            f_mean_hat = None
            s_2_hat = None
            print('Error! The type of mean ± std should be chosen from {L, G, LG}')

        s_hat = np.sqrt(s_2_hat)

        # step 2: LCB
        w = 2
        lcb_matrix = (f_mean_hat - w * s_hat).flatten()
        return lcb_matrix, -s_hat.flatten()

    @property
    def initialized(self):
        return self.client_model.trainable

    @property
    def x(self):
        return self.solutions.x

    @property
    def y(self):
        return self.solutions.y