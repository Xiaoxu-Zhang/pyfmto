from typing import Any

from ..core import ComponentData
from .problem import MultiTaskProblem, SingleTaskProblem
from .solution import Solution

__all__ = [
    'MultiTaskProblem',
    'ProblemData',
    'SingleTaskProblem',
    'Solution',
]


class ProblemData(ComponentData):
    problem: type[MultiTaskProblem]

    def __init__(self):
        self.params_basic: dict[str, Any] = {
            'npd': 1,
            'random_ctrl': 'weak',
            'seed': 123,
        }
        super().__init__()

    @property
    def available(self):
        return self._check_attr('problem', MultiTaskProblem)

    @property
    def params(self) -> dict[str, Any]:
        params = self._merged_params
        dim = params.get('dim', 0)
        if dim > 0:
            if 'fe_init' not in params:
                params.update(fe_init=5 * dim)
            if 'fe_max' not in params:
                params.update(fe_max=11 * dim)
        return params

    @property
    def name(self) -> str:
        name = super().name
        return name if self.name_suffix in name else f"{name}_{self.name_suffix}"

    def load_default_params(self):
        params_parsed = self._parse_default_params('problem')
        self.params_default.update(self.params_basic)
        self.params_default.update(params_parsed)

    def initialize(self) -> MultiTaskProblem:
        if self.available:
            return self.problem(**self.params)
        else:
            raise ValueError(f"Problem {self.name_orig} not available.")

    @property
    def name_suffix(self) -> str:
        return f"{self.task_num_str}_{self.dim_str}" if self.dim > 0 else self.task_num_str

    @property
    def npd(self) -> int:
        return self.params.get('npd', 0)

    @property
    def npd_str(self) -> str:
        return f"NPD{self.npd}" if self.npd > 0 else ""

    @property
    def dim(self) -> int:
        return self.params.get('dim', 0)

    @property
    def dim_str(self) -> str:
        if self.dim > 0:
            return f"{self.dim}D"
        else:
            return ""

    @property
    def task_num(self) -> int:
        if self.available:
            return len(self.problem(**{'_init_solutions': False}, **self.params))
        else:
            return 0

    @property
    def task_num_str(self) -> str:
        return f"{self.task_num}T"
