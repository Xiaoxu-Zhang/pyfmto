import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize


class ThompsonSampling:
    def __init__(self, x_lb, x_ub, dim, ts_trials):
        super().__init__(x_lb, x_ub, dim)
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.dim = dim
        if isinstance(x_lb, (int, float)):
            self.x_bound = (self.x_lb, self.x_ub)
        elif isinstance(x_lb, (list, tuple, np.ndarray)):
            self.x_bound = (self.x_lb[0], self.x_ub[0])
        else:
            raise ValueError(f"Type of x_lb is {type(x_lb)}, but it should be int or float or iterable")
        self.ts_trials = ts_trials

    def __call__(self, function, is_minimize, ts_trials=None, *args, **kwargs):
        if ts_trials is None:
            ts_trials = self.ts_trials
        # find the best x to maximize the acquisition function upon the entire bound
        # mimic the original implementation in DP-FTS-DE with full bound search
        x_tries = init_samples(self.x_lb, self.x_ub, dim=self.dim, size=ts_trials, kind='lhs')
        ac_values = function(x_tries)
        ac_best_idx = ac_values.argmax()
        ac_best = ac_values[ac_best_idx]
        x_best = x_tries[ac_best_idx]

        # L-BFGS-B optimization to find the best x
        x_seeds = init_samples(self.x_lb, self.x_ub, dim=self.dim, size=20, kind='lhs')
        assert len(x_seeds[0]) == self.dim
        for x_try in x_seeds:
            obj_fun = function if is_minimize else lambda x: -function(x)
            res = minimize(lambda x: obj_fun(x),
                           x_try,
                           bounds=[self.x_bound for _ in range(self.dim)],
                           method="L-BFGS-B")
            if ac_best is None or res.fun >= ac_best:
                x_best = res.x
                ac_best = res.fun
        return x_best, ac_best


def init_samples(x_lb, x_ub, dim, size, kind='lhs'):
    if kind == 'lhs':
        res = lhs(dim, samples=size)
    else:
        res = np.random.uniform(x_lb, x_ub, size=(size, dim))
    res = res * (x_ub - x_lb) + x_lb
    return res
