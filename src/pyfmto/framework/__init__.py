import textwrap
from pathlib import Path
from typing import Union, Literal

from .client import Client, record_runtime
from .server import Server
from .packages import ClientPackage, ServerPackage, DataArchive
from pyfmto.problems import load_problem
from pyfmto.algorithms import get_alg_kwargs
from pyfmto.utilities import colored, save_yaml, load_yaml
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

__all__ = [
    'Client',
    'Server',
    'DataArchive',
    'ClientPackage',
    'ServerPackage',
    'record_runtime',
    'export_alg_template',
    'export_launch_module',
    'export_default_config',
    'export_problem_config',
    'export_launcher_config',
    'export_reporter_config',
    'export_algorithm_config',
]


def export_launch_module():
    rows = """
    from pyfmto.experiments import Launcher


    if __name__ == '__main__':
        launcher = Launcher()
        launcher.run()
    """
    with open('run.py', 'w') as f:
        f.write(textwrap.dedent(rows))


def export_alg_template(name: str):
    alg_dir = Path(f'algorithms/{name.upper()}')
    alg_dir.mkdir(parents=True, exist_ok=True)
    clt_name = f"{name.title()}Client"
    srv_name = f"{name.title()}Server"
    clt_module = f"{name.lower()}_client"
    srv_module = f"{name.lower()}_server"
    clt_rows = f"""
        import time
        import numpy as np
        from pyfmto.framework import Client, record_runtime, ClientPackage, ServerPackage
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
        from pyfmto.framework import Server, ClientPackage, ServerPackage
        from pyfmto.utilities import logger\n\n
        class {srv_name}(Server):
            \"\"\"
            beta: 0.5
            \"\"\"
            def __init__(self, **kwargs):
                super().__init__()
                kwargs = self.update_kwargs(kwargs)
                self.beta = kwargs['beta']
            def handle_request(self, client_data: ClientPackage) -> ServerPackage:
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


def save_module(text, filename: Path):
    filename = filename.with_suffix('.py')
    if filename.exists():
        print(f"{colored('Skipped', 'yellow')} existing file {filename}")
        return
    with open(filename.with_suffix('.py'), 'w') as f:
        f.write(textwrap.dedent(text))
    print(f"{colored('Created', 'green')} {filename}")


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
    )
    return save_config('reporter', data.model_dump(), mode)


def export_algorithm_config(
        algs: tuple[str, ...] = ('BO', 'FMTBO'),
        mode: Literal['new', 'update'] = 'new'
):
    all_algorithms = {}
    for name in algs:
        try:
            curr = get_alg_kwargs(name)
            all_algorithms[name] = curr
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


def save_config(name: str, data, mode: Literal['new', 'update'] = 'new'):
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


def export_default_config():
    export_launcher_config(mode='update')
    export_reporter_config(mode='update')
    export_algorithm_config(mode='update')
    export_problem_config(mode='update')
