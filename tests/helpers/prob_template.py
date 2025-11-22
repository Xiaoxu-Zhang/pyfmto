from typing import Union

import numpy as np
from pyfmto.problems import SingleTaskProblem, MultiTaskProblem


class StpA(SingleTaskProblem):

    def __init__(self, dim: int):
        super().__init__(dim=dim, obj=1, lb=-1, ub=1)

    def _eval_single(self, x: np.ndarray):
        return np.sum(x ** 2)


class StpB(SingleTaskProblem):

    def __init__(self, dim: int):
        super().__init__(dim=dim, obj=1, lb=-2, ub=2)

    def _eval_single(self, x: np.ndarray):
        return np.sin(sum(x))


class StpC(SingleTaskProblem):

    def __init__(self, dim: int):
        super().__init__(dim=dim, obj=1, lb=-3, ub=3)

    def _eval_single(self, x: np.ndarray):
        return np.cos(sum(x))


class StpD(SingleTaskProblem):
    def __init__(self, dim: int):
        super().__init__(dim=dim, obj=1, lb=-4, ub=4)

    def _eval_single(self, x: np.ndarray):
        return np.tan(sum(x))


class NameProb(MultiTaskProblem):

    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs) -> Union[list[SingleTaskProblem], tuple[SingleTaskProblem, ...]]:
        return StpA(dim), StpB(dim), StpC(dim), StpD(dim)
