import numpy as np
from smt.surrogate_models import GPX
from scipy.stats import norm

from pyfmto.framework import Client, record_runtime, ClientPackage, Actions, ServerPackage
from .bo_utils import ThompsonSampling


class BoClient(Client):
    """
    ts_trials: 1000
    num_seeds: 20
    sync: true
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        kwargs = self.update_kwargs(kwargs)
        self.ts_trials = kwargs['ts_trials']
        self.num_seeds = kwargs['num_seeds']
        self.sync = kwargs['sync']
        self.gp = GPX(theta0=[1e-2], print_global=False)
        self.ts = ThompsonSampling(self.x_lb, self.x_ub, self.dim, ts_trials=self.ts_trials, num_seeds=self.num_seeds)
        self.sync_ver = 1

    @property
    def x(self) -> np.ndarray:
        return self.solutions.x

    @property
    def y(self) -> np.ndarray:
        return self.solutions.y

    def optimize(self):
        self.fit()
        x = self.optimize_acquisition()
        y = self.problem.evaluate(x)
        self.solutions.append(x.reshape(1, -1), y.reshape(1, -1))
        self.sync_version()

    @record_runtime('Waiting version sync')
    def sync_version(self):
        # Synchronize the progress state across all clients
        if self.sync:
            self.push()
            self.pull()
            self.sync_ver += 1

    def push(self):
        pkg = ClientPackage(self.id, Actions.PUSH_UPDATE, self.sync_ver)
        self.request_server(pkg)

    def pull(self):
        pkg = ClientPackage(self.id, Actions.PULL_UPDATE, self.sync_ver)
        self.request_server(pkg, repeat=100)

    def check_pkg(self, x: ServerPackage) -> bool:
        if x is None:
            return False

        if isinstance(x.data, str):
            return True
        else:
            return x.data >= self.sync_ver

    @record_runtime("Fit")
    def fit(self):
        self.gp.set_training_values(self.x, self.y)
        self.gp.train()

    def predict(self, x, return_mse=False):
        mean = self.gp.predict_values(x)
        if return_mse:
            mse = self.gp.predict_variances(x)
            return mean, mse
        return mean

    @record_runtime("Acq")
    def optimize_acquisition(self):
        return self.ts.minimize(lambda x: -self.ei(x))

    def ei(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean_y_new, sigma_y_new = self.predict(x, return_mse=True)
        sigma_y_new[sigma_y_new == 0] = 1e-15
        z = (self.y_min - mean_y_new) / sigma_y_new
        exp_imp = (self.y_min - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return exp_imp.flatten()