import copy
import numpy as np
from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant
from scipy.stats import norm
from typing import Union
from pyfmto.framework import Client, ClientPackage, Actions, ServerPackage, record_runtime
from pyfmto.utilities.tools import update_kwargs

from .fmtbo_utils import GeneticAlgorithm, AggData

rng = np.random.default_rng()


class FmtboClient(Client):
    """
    gamma: 0.5
    max_gen: 20
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        kwargs = update_kwargs('FmtboClient', self.default_kwargs(), kwargs)

        # init control args
        self.gamma = kwargs['gamma']

        # init model args
        self.d_share = None
        self.global_gp_params = None
        self.global_gp_model = None
        self.local_gp_model = None

        self._ga_operator = GeneticAlgorithm(x_lb=self.x_lb, x_ub=self.x_ub, dim=self.dim,
                                             pop_size=self.fe_init,
                                             max_gen=kwargs['max_gen'])

        self.prev_ver = 0

    @property
    def global_model_params(self):
        if self.global_gp_model is not None:
            return self.global_gp_model.model['theta']

    @property
    def local_model_params(self):
        if self.local_gp_model is not None:
            return self.local_gp_model.model['theta']

    def optimize(self):
        if self.d_share is None:
            self._initializing()
        else:
            self._optimizing()

    def _initializing(self):
        pkg = ClientPackage(
            cid=self.id, action=Actions.PUSH_INIT,
            data={'dim': self.dim, 'bound': (self.x_lb,self.x_ub), 'd_share_size': self.fe_init})
        self.request_server(package=pkg)
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_INIT, data=None)
        server_pkg: ServerPackage = self.request_server(package=pkg)
        self.d_share = server_pkg.data['d_share']
        self.global_gp_params = server_pkg.data['theta']
        self._init_gp_model('global', self.global_gp_params)
        self._init_gp_model('local', 5)
        self.local_gp_model.fit(self.solutions.x, self.solutions.y)
        self.global_gp_model.fit(self.solutions.x, self.solutions.y)

    @record_runtime(name='Opt')
    def _optimizing(self):
        self.push()
        self.pull()
        self._update_surrogate()
        x_next = self._find_next_x()
        y_next = self.problem.evaluate(x_next)
        self.solutions.append(x_next.reshape(-1, self.dim), y_next.reshape(-1, self.obj))

    @record_runtime(name='Push')
    def push(self):
        data = {'rank': self._cal_x_s_share_rank(),
                'size': self._local_data_size,
                'global': self.global_model_params}
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_UPDATE, data=data)
        self.request_server(package=pkg, msg="Push update")

    @record_runtime(name='Pull')
    def pull(self):
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, data=None)
        self.record_round_info('DataVer', f"ver {self.prev_ver}")
        res = self.request_server(
            package=pkg,
            repeat=100,
            interval=1,
            msg=f"Pull update, require Ver{self.prev_ver+1}")
        self.prev_ver = res.data.version
        self.global_gp_params = res.data.agg_res

    def check_pkg(self, x) -> bool:
        if x:
            if x.data is None:
                return False
            elif isinstance(x.data, AggData):
                return x.data.version > self.prev_ver
            else:
                return True
        else:
            return False

    def _synchronize_gp_params(self, theta:Union[int, float, np.ndarray]):
        self._init_gp_model('global', theta)

    def _init_gp_model(self, model_type, theta:Union[int, float, np.ndarray]):
        if isinstance(theta, (int, float)):
            theta = theta * np.ones(self.dim)
        else:
            assert theta.shape[0] == self.dim, f"Theta shape[0]={theta.shape[0]} != dim={self.dim}"
        model = DACE(
            regr=regr_constant,
            corr=corr_gauss,
            theta=theta,
            thetaL=1e-5 * np.ones(self.dim),
            thetaU=100 * np.ones(self.dim))

        if model_type == "global":
            self.global_gp_model = model
        elif model_type == "local":
            self.local_gp_model = model
        else:
            raise ValueError("model_type should be 'global' or 'local'")

    def _check_new_x(self, x_new):
        x_next_best = x_new
        if self.solutions.size != 0:
            if x_new is None:
                x_next_best = self.problem.random_uniform_x(1)
            elif np.any(np.all(self.solutions.x - x_new == 0, axis=1)):
                x_next_best = self.problem.random_uniform_x(1)
        return x_next_best

    @record_runtime(name='GA')
    def _ga_method(self, no_generations=20):
        archive = copy.deepcopy(self.solutions.x)
        return self._ga_operator.optimize(function=self._acq_eval, archive=archive, max_gen=no_generations)

    def _acq_eval(self, x_new):
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(1, -1)
        return self._eval_x_on_ei_w(x_new)

    def _eval_x_on_ei_w(self, x_new):
        mean_y_new, var_y_new = self.global_gp_model.predict(x_new, return_mse=True)
        l_mean_y_new, l_var_y_new = self.local_gp_model.predict(x_new, return_mse=True)
        sigma_y_new = np.sqrt(np.abs(var_y_new))
        l_sigma_y_new = np.sqrt(np.abs(l_var_y_new))
        return self._EI_w(mean_y_new=mean_y_new, sigma_y_new=sigma_y_new,
                          l_mean_y_new=l_mean_y_new, l_sigma_y_new=l_sigma_y_new)

    def _EI_w(self, mean_y_new, sigma_y_new, l_mean_y_new, l_sigma_y_new):
        sigma_y_new[sigma_y_new == 0] = 1e-15
        l_sigma_y_new[l_sigma_y_new == 0] = 1e-15

        min_y = self.y_min
        z = (min_y - mean_y_new) / sigma_y_new
        exp_imp = (min_y - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        l_z = (min_y - l_mean_y_new) / l_sigma_y_new
        l_exp_imp = (min_y - l_mean_y_new) * norm.cdf(l_z) + l_sigma_y_new * norm.pdf(l_z)
        exp_imp = self.gamma * exp_imp + (1 - self.gamma) * l_exp_imp
        return np.squeeze(exp_imp)

    def _find_next_x(self):
        next_x, _ = self._ga_method(no_generations=20)
        self._check_new_x(next_x)
        return next_x

    def _update_surrogate(self) -> None:
        self._synchronize_gp_params(self.global_gp_params)
        self.global_gp_model.fit(X=self.solutions.x, Y=self.solutions.y)
        self.local_gp_model.fit(X=self.solutions.x, Y=self.solutions.y)

    def _cal_x_s_share_rank(self):
        mean_ys_share = self.global_gp_model.predict(self.d_share, return_mse=False)
        # ascend ranking
        rank_idx = np.squeeze(mean_ys_share).argsort()
        # representing ascend rank of predictions
        return rank_idx.argsort()

    @property
    def _local_data_size(self):
        try:
            return self.problem.data_size
        except Exception:
            return 1