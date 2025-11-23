import numpy as np
from pathlib import Path
from typing import Union
from pyfmto.experiments import RunSolutions
from pyfmto.experiments.utils import MergedResults, MetaData
from pyfmto.problems import Solution
from pyfmto.utilities.schemas import STPConfig


__all__ = ['gen_algorithm', 'gen_problem', 'ExpDataGenerator']


def gen_algorithm(names: Union[str, list[str]]):
    name_lst = names if isinstance(names, list) else [names]
    for name in name_lst:
        alg_dir = Path('algorithms') / name
        alg_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(__file__).parent / 'alg_template.py', 'r') as f1:
            template = f1.read().replace('NameAlg', name.title())
            with open(alg_dir / '__init__.py', 'w') as f2:
                f2.write(template)
        print(f"Successfully generated algorithm {name}")


def gen_problem(names: Union[str, list[str]]):
    from .cleaners import clear_problems_cache
    clear_problems_cache()
    name_lst = names if isinstance(names, list) else [names]
    for name in name_lst:
        root = Path('problems')
        root.mkdir(parents=True, exist_ok=True)
        with open(Path(__file__).parent / 'prob_template.py', 'r') as f1:
            template = f1.read().replace('NameProb', name)
            with open(root / f"{name.lower()}.py", 'w') as f2:
                f2.write(template)
            with open(root / '__init__.py', 'a') as f3:
                f3.write(f"from .{name.lower()} import {name}\n")
        print(f"Successfully generated problem {name}")


class ExpDataGenerator:
    def __init__(self, dim: int, lb: float, ub: float):
        self.dim = dim
        self.lb, self.ub = lb, ub
        self.conf = STPConfig(dim=dim, obj=1, lb=lb, ub=ub)

    def gen_solutions(self, n_solutions: int) -> list[Solution]:
        res = [self.gen_solution() for _ in range(n_solutions)]
        return res

    def gen_solution(self):
        sol = Solution(self.conf)
        x = np.random.uniform(self.lb, self.ub, size=(sol.fe_max, self.dim))
        y = np.random.uniform(1e-5, self.ub, size=(sol.fe_max, 1))
        sol.append(x, y)
        sol._x_global = (self.conf.lb + self.conf.ub) / 2
        sol._y_global = 0.0
        return sol

    def gen_run_data(self, n_tasks: int) -> RunSolutions:
        run_data = RunSolutions()
        for tid in range(n_tasks):
            run_data.update(tid+1, self.gen_solution())
        return run_data

    def gen_runs_data(self, n_tasks: int, n_runs: int) -> list[RunSolutions]:
        return [self.gen_run_data(n_tasks) for _ in range(n_runs)]

    def gen_merged_data(self, n_tasks: int, n_runs: int):
        return MergedResults(self.gen_runs_data(n_tasks, n_runs))

    def gen_metadata(self, algs: list[str], prob: str, npd: str, n_tasks: int, n_runs: int):
        data = {alg: self.gen_merged_data(n_tasks, n_runs) for alg in algs}
        return MetaData(data, prob, npd, Path('tmp/reports'))
