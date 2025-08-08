"""
Single-Task Single-Objective Benchmark Problems

This module implements a collection of classical single-objective continuous optimization benchmark functions,
commonly used in evolutionary multitasking optimization, federated optimization, and other multitask learning scenarios.

Each function is defined as a class derived from the base class `SingleTaskProblem`, and overrides the `evaluate` method
to support batched input and vectorized computation using NumPy.

The following benchmark problems are included:
    - Griewank
    - Rastrigin
    - Ackley
    - Schwefel
    - Sphere
    - Rosenbrock
    - Weierstrass
    - Ellipsoid

References
----------
Definition:
    B. Da, Y.-S. Ong, L. Feng, A. K. Qin, A. Gupta, Z. Zhu, C.-K. Ting,
    K. Tang, and X. Yao, "Evolutionary multitasking for single-objective
    continuous optimization: Benchmark problems, performance metric, and
    baseline results," arXiv preprint arXiv:1706.03470, 2017.

Implementation:
    Zhu, H., Wang, X., & Jin, Y. (2023). Federated Many-Task Bayesian
    Optimization. IEEE Transactions on Evolutionary Computation, 1–1.
    https://doi.org/10.1109/TEVC.2023.3279775
"""

import numpy as np

from ..problem import SingleTaskProblem as Stp

__all__ = [
    "Ackley",
    "Sphere",
    "Schwefel",
    "Griewank",
    "Ellipsoid",
    "Rastrigin",
    "Rosenbrock",
    "Weierstrass",
]


class Griewank(Stp):

    def __init__(self, dim=10, lb=-600, ub=600, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        i = np.arange(self.dim) + 1
        f1 = np.sum(x ** 2 / 4000)
        f2 = np.prod(np.cos(x / np.sqrt(i)))
        out = f1 - f2 + 1
        return out


class Rastrigin(Stp):

    def __init__(self, dim=10, lb=-5, ub=5, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        square_term = x ** 2
        cos_term = 10 * np.cos(2 * np.pi * x)
        out = np.sum(square_term - cos_term + 10)
        return out


class Ackley(Stp):

    def __init__(self, dim=10, lb=-32.768, ub=32.768, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        out = -20 * np.exp(-0.2 * np.sqrt(sum1 / self.dim)) - np.exp(sum2 / self.dim) + 20 + np.e
        return out


class Schwefel(Stp):

    def __init__(self, dim=10, lb=-500, ub=500, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)
        self.set_x_global(np.ones(self.dim) * 420.9687)

    def _eval_single(self, x):
        term = x * np.sin(np.sqrt(np.abs(x)))
        out = 418.9829 * self.dim - np.sum(term)
        return out


class Sphere(Stp):

    def __init__(self, dim=10, lb=-100, ub=100, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        return np.sum(x ** 2)


class Rosenbrock(Stp):
    def __init__(self, dim=10, lb=-2.048, ub=2.048, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)
        self.set_x_global(np.ones(self.dim))

    def _eval_single(self, x):
        x_squared_diff = x[1:] ** 2 - x[:-1]
        term1 = 100 * np.sum(np.power(x_squared_diff, 2))
        term2 = np.sum(np.power(x[:-1] - 1, 2))
        return term1 + term2


class Weierstrass(Stp):

    def __init__(self, dim=10, lb=-0.5, ub=0.5, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        a = 0.5
        b = 3
        kmax = 21

        k_values = np.arange(kmax)
        a_k = a ** k_values
        b_k = b ** k_values

        term1 = np.sum(a_k * np.cos(2 * np.pi * b_k * (x[:, np.newaxis] + 0.5)), axis=1)
        term2 = np.sum(a_k * np.cos(2 * np.pi * b_k * 0.5), axis=0)

        obj = np.sum(term1) - self.dim * term2

        return obj


class Ellipsoid(Stp):

    def __init__(self, dim=10, lb=-5.12, ub=5.12, **kwargs):
        super().__init__(dim=dim, obj=1, lb=lb, ub=ub, **kwargs)

    def _eval_single(self, x):
        d_l = np.arange(self.dim) + 1
        out = np.sum(d_l * x ** 2)
        return out
