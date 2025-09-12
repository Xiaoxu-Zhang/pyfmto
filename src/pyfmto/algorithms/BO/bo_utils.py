from typing import Any

import numpy as np
from enum import Enum, auto
from pyDOE import lhs
from scipy.optimize import minimize
from pyfmto.framework import ClientPackage as Pkg


class ClientPackage(Pkg):
    def __init__(self, cid: int, action: Enum, data: Any = None):
        super().__init__(cid, action)
        self.data = data


class Actions(Enum):
    PUSH_UPDATE = auto()
    PULL_UPDATE = auto()


class ThompsonSampling:
    def __init__(self, x_lb, x_ub, dim, ts_trials, num_seeds):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.dim = dim
        self.num_seeds = num_seeds
        self.ts_trials = ts_trials
        if isinstance(x_lb, (int, float)):
            self.bounds = [(self.x_lb, self.x_ub) for _ in range(self.dim)]
        elif isinstance(x_lb, (list, tuple, np.ndarray)):
            self.bounds = [(self.x_lb[i], self.x_ub[i]) for i in range(self.dim)]
        else:
            raise ValueError(f"Type of x_lb is {type(x_lb)}, but it should be int or float or iterable")

    def minimize(self, function):
        x_tries = init_samples(self.x_lb, self.x_ub, dim=self.dim, size=self.ts_trials)
        ac_values = function(x_tries)
        ac_min_idx = ac_values.argmin()
        ac_min = ac_values[ac_min_idx]
        ac_min_x = x_tries[ac_min_idx]

        # L-BFGS-B optimization to find the best x
        x_seeds = init_samples(self.x_lb, self.x_ub, dim=self.dim, size=self.num_seeds)
        for x_try in x_seeds:
            res = minimize(function,
                           x_try,
                           bounds=self.bounds,
                           method="L-BFGS-B"
                           )
            if ac_min is None or res.fun < ac_min:
                ac_min_x = res.x
                ac_min = res.fun
        return ac_min_x


def init_samples(lb, ub, dim, size):
    return lhs(dim, samples=size) * (ub - lb) + lb
