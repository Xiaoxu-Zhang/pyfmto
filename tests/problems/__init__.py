import numpy as np

from pyfmto.problems import SingleTaskProblem, MultiTaskProblem as _MultiTaskProblem

TASK_NUM = 4


class ConstantProblem(SingleTaskProblem):
    def _eval_single(self, x):
        return np.zeros(self.obj)


class SimpleProblem(SingleTaskProblem):
    def _eval_single(self, x):
        return np.sin(np.sum(x ** 2)) * np.ones(self.obj)


class MtpNonIterableReturn(_MultiTaskProblem):
    is_realworld = False

    def _init_tasks(self, **kwargs):
        return 0


class MtpSynthetic(_MultiTaskProblem):
    is_realworld = False

    def _init_tasks(self, **kwargs):
        return [SimpleProblem(2, 1, 0, 1, **kwargs) for _ in range(TASK_NUM)]


class MtpRealworld(_MultiTaskProblem):
    is_realworld = True

    def _init_tasks(self, **kwargs):
        return [SimpleProblem(2, 1, 0, 1, **kwargs) for _ in range(TASK_NUM)]
