import importlib
from pathlib import Path
from typing import Optional
from opfunu import draw_3d
import numpy as np
from pyfmto.problems import SingleTaskProblem as Stp, MultiTaskProblem as Mtp

__all__ = ['Cec2022']


class _BenchmarksCec2022(Stp):
    def __init__(self, fid, dim: int, **kwargs):
        cec2022 = importlib.import_module('opfunu.cec_based.cec2022')
        func = getattr(cec2022, f"F{fid}2022")(dim)
        self.func = func
        super().__init__(dim, 1, x_lb=func.lb, x_ub=func.ub, **kwargs)

    def evaluate(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        res = map(self._eval_one, x)
        return np.array(list(res)).reshape(-1, 1)

    def visualize(self, filename: Optional[str] = None, num_points=300):
        if filename is None:
            draw_3d(self.func.evaluate, self.x_lb, self.x_ub, n_points=num_points)
        else:
            filename = Path(filename).with_suffix('.png') if filename is not None else None
            draw_3d(self.func.evaluate, self.x_lb, self.x_ub, exts=('', ), filename=str(filename), n_points=num_points, verbose=False)

    def _eval_one(self, x):
         return self.func.evaluate(x)


class Cec2022(Mtp):
    def __init__(self, dim: int, **kwargs):
        if not dim in [10, 20]:
            raise ValueError('CEC2022 only support 10D and 20D')
        super().__init__(False, dim, **kwargs)

    def _init_tasks(self, dim, **kwargs):
        funcs = [_BenchmarksCec2022(fid, dim, **kwargs) for fid in range(1, 13)]
        return funcs
