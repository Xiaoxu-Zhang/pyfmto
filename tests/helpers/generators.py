import numpy as np
from pathlib import Path
from typing import Union, Literal
from pyfmto import load_problem
from pyfmto.experiments import RunSolutions
from pyfmto.experiments.utils import MergedResults, MetaData
from pyfmto.problems import Solution
from pyfmto.utilities import save_yaml, load_yaml
from pyfmto.utilities.loaders import LauncherConfig, ReporterConfig
from pyfmto.utilities.schemas import STPConfig


def gen_algorithm(name: str):
    alg_dir = Path('algorithms') / name
    alg_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent / 'alg_template.py', 'r') as f1:
        template = f1.read().replace('NameAlg', name.title())
        with open(alg_dir / '__init__.py', 'w') as f2:
            f2.write(template)


def gen_problem(name: str):
    root = Path('problems')
    root.mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).parent / 'prob_template.py', 'r') as f1:
        template = f1.read().replace('NameProb', name)
        with open(root / f"{name.lower()}.py", 'w') as f2:
            f2.write(template)
        with open(root / '__init__.py', 'a') as f3:
            f3.write(f"from .{name.lower()} import {name}\n")
    print(f'Problem {name} created successfully.')


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


def save_config(name: str, data, mode: str = 'new'):
    if mode == 'new':
        number = 1
        path = Path(f'config_{name}{number}.yaml')
        while True:
            if not path.exists():
                break
            number += 1
            path = Path(f'config_{name}{number}.yaml')
        if data:
            save_yaml({name: data}, path)
            return Path(f'config_{name}{number}.yaml')
    else:
        try:
            old = load_yaml('config.yaml')
        except FileNotFoundError:
            old = {}
        if data:
            old.update({name: data})
        key_order = ['launcher', 'reporter', 'algorithms', 'problems']
        old = {k: old[k] for k in key_order if k in old}
        save_yaml(old, 'config.yaml')
        return Path('config.yaml')


def export_launcher_config(
        results: str = 'out/results',
        repeat: int = 1,
        save: bool = True,
        seed: int = 42,
        backup: bool = True,
        algs: Union[tuple[str, ...], list[str]] = ('BO', 'FMTBO'),
        probs: Union[tuple[str, ...], list[str]] = ('tetci2019', 'arxiv2017'),
        mode: Literal['new', 'update'] = 'new',
):
    data = LauncherConfig(
        results=results,
        repeat=repeat,
        save=save,
        seed=seed,
        backup=backup,
        algorithms=algs,
        problems=probs,
    )
    return save_config('launcher', data.model_dump(), mode)


def export_reporter_config(
        results: str = 'out/results',
        algs: tuple[list[str], ...] = (['BO', 'FMTBO'], ),
        probs: tuple[str, ...] = ('tetci2019', 'arxiv2017'),
        mode: Literal['new', 'update'] = 'new',
):
    data = ReporterConfig(
        results=results,
        algorithms=list(algs),
        problems=list(probs),
        formats=['curve']
    )
    return save_config('reporter', data.model_dump(), mode)


def export_algorithm_config(
        algs: tuple[str, ...],
        mode: Literal['new', 'update'] = 'new'
):
    all_algorithms = {}
    for name in algs:
        try:
            from pyfmto import load_algorithm
            curr = load_algorithm(name)
            all_algorithms[name] = curr.params_default
        except Exception as e:
            print(e)
    return save_config('algorithms', all_algorithms, mode)


def export_problem_config(
        probs: tuple[str, ...] = ('arxiv2017', 'tetci2019'),
        mode: Literal['new', 'update'] = 'new'
):
    all_problems = {}
    for name in probs:
        try:
            load_problem(name)
            all_problems[name] = {
                'dim': 10,
                'seed': 123,
                'np_per_dim': 1,
                'random_ctrl': 'weak',
            }
        except Exception as e:
            print(e)
    return save_config('problems', all_problems, mode)
