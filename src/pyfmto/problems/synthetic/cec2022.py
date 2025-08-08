import importlib
from pyfmto.problems import SingleTaskProblem as Stp, MultiTaskProblem as Mtp

__all__ = ['Cec2022']


class _BenchmarksCec2022(Stp):
    def __init__(self, fid, dim: int, **kwargs):
        cec2022 = importlib.import_module('opfunu.cec_based.cec2022')
        func = getattr(cec2022, f"F{fid}2022")(dim)
        super().__init__(dim, 1, lb=func.lb, ub=func.ub, **kwargs)
        self.func = func
        self.set_x_global(func.x_global)

    @property
    def name(self) -> str:
        return f"CEC2022F{self.id}"

    def _eval_single(self, x):
        return self.func.evaluate(x)


class Cec2022(Mtp):
    is_realworld = False
    intro = """
        Synthetic of CEC 2022 benchmark problems
    """

    def __init__(self, dim=10, **kwargs):
        if dim not in [10, 20]:
            raise ValueError('CEC2022 only support 10D and 20D')
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs):
        funcs = [_BenchmarksCec2022(fid, dim, **kwargs) for fid in range(1, 13)]
        return funcs
