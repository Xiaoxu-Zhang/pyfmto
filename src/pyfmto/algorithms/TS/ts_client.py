import numpy as np
from pydacefit.dace import DACE
from pydacefit.corr import corr_gauss
from pydacefit.regr import regr_constant
from scipy.stats import norm

from pyfmto.framework import Client, record_runtime
from .ts_utils import ThompsonSampling


class FtsClient(Client):
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.update_kwargs(kwargs)
        self.gp = DACE(regr=regr_constant, corr=corr_gauss)
        self.ts = ThompsonSampling(self.x_lb, self.x_ub, self.dim, ts_trials=100)

    @property
    def x(self) -> np.ndarray:
        return self.solutions.x

    @property
    def y(self) -> np.ndarray:
        return self.solutions.y

    def optimize(self):
        self.fitting()
        x = self.acquiring()
        y = self.problem.evaluate(x)
        self.solutions.append(x.reshape(1, -1), y.reshape(1, -1))

    @record_runtime("Fit")
    def fitting(self):
        self.gp.fit(self.x, self.y)

    @record_runtime("Acq")
    def acquiring(self):
        return self.ts.minimize(self.acq_ei)

    def acq_ei(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean_y_new, sigma_y_new = self.gp.predict(x, return_mse=True)
        sigma_y_new[sigma_y_new == 0] = 1e-15
        z = (self.y_min - mean_y_new) / sigma_y_new
        exp_imp = (self.y_min - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return exp_imp.flatten()