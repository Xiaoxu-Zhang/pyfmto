import numpy as np
import textwrap
from pathlib import Path
from typing import Union, Literal
from pyfmto import load_problem
from pyfmto.utilities import save_yaml, load_yaml
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig
from pyfmto.experiments import RunSolutions
from pyfmto.experiments.utils import MergedResults, MetaData
from pyfmto.problems import Solution
from pyfmto.utilities.schemas import STPConfig


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


def save_module(text, filename: Path):
    filename = filename.with_suffix('.py')
    with open(filename.with_suffix('.py'), 'w') as f:
        f.write(textwrap.dedent(text))


def export_alg_template(name: str):
    """
    Export algorithm template files for client and server components.

    Creates a new algorithm directory with client and server implementation
    templates, including basic structure and placeholder methods.

    Parameters
    ----------
    name : str
        Name of the algorithm to create template for.
    """
    alg_dir = Path(f'algorithms/{name.upper()}')
    alg_dir.mkdir(parents=True, exist_ok=True)
    clt_name = f"{name.title()}Client"
    srv_name = f"{name.title()}Server"
    clt_module = f"{name.lower()}_client"
    srv_module = f"{name.lower()}_server"
    clt_rows = f"""
        import time
        import numpy as np
        from pyfmto.framework import Client, record_runtime, ClientPackage
        from pyfmto.utilities import logger\n\n
        class {clt_name}(Client):
            \"\"\"
            alpha: 0.02
            \"\"\"
            def __init__(self, problem, **kwargs):
                super().__init__(problem)
                kwargs = self.update_kwargs(kwargs)
                self.alpha = kwargs['alpha']
                self.problem.auto_update_solutions = True\n
            def optimize(self):
                x = self.problem.random_uniform_x(1)
                self.problem.evaluate(x)
                time.sleep(self.alpha)\n
    """

    srv_rows = f"""
        from pyfmto.framework import Server, ClientPackage
        from pyfmto.utilities import logger\n\n
        class {srv_name}(Server):
            \"\"\"
            beta: 0.5
            \"\"\"
            def __init__(self, **kwargs):
                super().__init__()
                kwargs = self.update_kwargs(kwargs)
                self.beta = kwargs['beta']
            def handle_request(self, client_data: ClientPackage):
                pass
            def aggregate(self):
                pass
    """

    init_rows = f"""
        from .{srv_module} import {srv_name}
        from .{clt_module} import {clt_name}
    """

    save_module(srv_rows, alg_dir / srv_module)
    save_module(clt_rows, alg_dir / clt_module)
    save_module(init_rows, alg_dir / '__init__')


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
    """
    Export launcher configuration to YAML file.

    Creates or updates launcher configuration with experiment settings
    including results path, repetition count, algorithms and problems.

    Parameters
    ----------
    results : str, optional
        Path to store results (default is 'out/results')
    repeat : int, optional
        Number of repetitions for experiments (default is 1)
    save : bool, optional
        Whether to save results (default is True)
    seed : int, optional
        Random seed for experiments (default is 42)
    backup : bool, optional
        Whether to backup previous results (default is True)
    algs : Union[tuple[str, ...], list[str]], optional
        List of algorithm names to include (default is ('BO', 'FMTBO'))
    probs : Union[tuple[str, ...], list[str]], optional
        List of problem names to include (default is ('tetci2019', 'arxiv2017'))
    mode : Literal['new', 'update'], optional
        Export mode, either 'new' or 'update' (default is 'new')

    Returns
    -------
    Path
        Path to the created or updated configuration file.
    """
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
    """
    Export reporter configuration to YAML file.

    Creates or updates reporter configuration with results path,
    algorithms and problems for generating reports.

    Parameters
    ----------
    results : str, optional
        Path to results data (default is 'out/results')
    algs : tuple[list[str], ...], optional
        Nested list of algorithm names for comparison (default is (['BO', 'FMTBO'], ))
    probs : tuple[str, ...], optional
        List of problem names to include in reports (default is ('tetci2019', 'arxiv2017'))
    mode : Literal['new', 'update'], optional
        Export mode, either 'new' or 'update' (default is 'new')

    Returns
    -------
    Path
        Path to the created or updated configuration file.
    """
    data = ReporterConfig(
        results=results,
        algorithms=list(algs),
        problems=list(probs),
    )
    return save_config('reporter', data.model_dump(), mode)


def export_algorithm_config(
        algs: tuple[str, ...],
        mode: Literal['new', 'update'] = 'new'
):
    """
    Export algorithm parameters configuration to YAML file.

    Creates or updates algorithm configuration with parameter settings
    for specified algorithms.

    Parameters
    ----------
    algs : tuple[str, ...], optional
        List of algorithm names to configure (default is ('BO', 'FMTBO'))
    mode : Literal['new', 'update'], optional
        Export mode, either 'new' or 'update' (default is 'new')

    Returns
    -------
    Path
        Path to the created or updated configuration file.
    """

    all_algorithms = {}
    for name in algs:
        try:
            from pyfmto.experiments.loaders import get_alg_kwargs
            curr = get_alg_kwargs(name)
            all_algorithms[name] = curr
        except Exception as e:
            print(e)
    return save_config('algorithms', all_algorithms, mode)


def export_problem_config(
        probs: tuple[str, ...] = ('arxiv2017', 'tetci2019'),
        mode: Literal['new', 'update'] = 'new'
):
    """
    Export problem configuration to YAML file.

    Creates or updates problem configuration with dimension, seed
    and other problem-specific settings.

    Parameters
    ----------
    probs : tuple[str, ...], optional
        List of problem names to configure (default is ('arxiv2017', 'tetci2019'))
    mode : Literal['new', 'update'], optional
        Export mode, either 'new' or 'update' (default is 'new')

    Returns
    -------
    Path
        Path to the created or updated configuration file.
    """
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
