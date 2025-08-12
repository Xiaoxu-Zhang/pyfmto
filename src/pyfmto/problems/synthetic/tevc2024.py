from pathlib import Path
from scipy.io import loadmat
from typing import Literal, Type

from .. import benchmarks
from ..problem import (MultiTaskProblem as Mtp, SingleTaskProblem)

__all__ = ["Tevc2024"]

T_SrcProblem = Literal[
    'Griewank', 'Rastrigin', 'Ackley', 'Schwefel', 'Sphere', 'Rosenbrock', 'Weierstrass', 'Ellipsoid']


class Tevc2024(Mtp):
    is_realworld = False
    intro = """
        This module implements a multi-task optimization benchmark derived from a single base function,
        with each task being transformed using different rotation matrices. The tasks share the same
        dimensionality and search space bounds, but differ in their problem landscapes due to rotations.
        The number of tasks is fixed at 10. Each task is a transformed version of the specified source problem.
    """
    notes = """
        All tasks are derived from the same base function.
        Tasks are differentiated by applying distinct rotation matrices.
        No shift transformation is applied.
    """
    references = [
        """
        Wang, X., & Jin, Y. (2024). Distilling Ensemble Surrogates for Federated Data-Driven
        Many-Task Optimization. IEEE Transactions on Evolutionary Computation, 1–1.
        https://doi.org/10.1109/TEVC.2024.3428701
        """
    ]

    def __init__(self, dim=10, src_problem: T_SrcProblem = 'Ackley', **kwargs):
        if not isinstance(src_problem, str):
            raise TypeError(f"original_problem should be str, but {type(src_problem)} is given")
        try:
            src_prob_cls = getattr(benchmarks, src_problem)
        except AttributeError:
            raise ValueError(f"{src_problem} is not exist, supported names "
                             f"[Griewank, Rastrigin, Ackley, Schwefel, Sphere"
                             f", Rosenbrock, Weierstrass, Ellipsoid]")
        super().__init__(dim, src_prob_cls, **kwargs)

    def get_info(self):
        task = self._problem[0]
        return {
            "TaskSrc": [task.name],
            "DecDim": [task.dim],
            "Lower": [task.lb[0]],
            "Upper": [task.ub[0]]
        }

    def _init_tasks(self, dim: int, src_prob_cls: Type[SingleTaskProblem], **kwargs):
        funcs = [src_prob_cls(dim, **kwargs) for _ in range(10)]
        datasets = Path(__file__).parents[1] / 'datasets' / 'tevc2024' / 'composition_func_M_D10.mat'
        rot_mats = loadmat(str(datasets))
        mats = [rot_mats[f"M{i + 1}"][0:dim, 0:dim] for i in range(10)]
        for f, mat in zip(funcs, mats):
            f.set_transform(rotation=mat, shift=None)
        return funcs
