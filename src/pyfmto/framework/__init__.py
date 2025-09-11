import textwrap
from pathlib import Path
from typing import Union, Literal

from .client import Client, record_runtime
from .server import Server
from .packages import ClientPackage, ServerPackage, SyncDataManager, DataArchive
from pyfmto.problems import load_problem
from pyfmto.algorithms import get_alg_kwargs
from pyfmto.utilities import colored, save_yaml, load_yaml
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

__all__ = [
    'Client',
    'Server',
    'DataArchive',
    'SyncDataManager',
    'ClientPackage',
    'ServerPackage',
    'record_runtime',
    'export_demo',
    'export_alg_template',
    'export_launch_module',
    'export_default_config',
    'export_problem_config',
    'export_launcher_config',
    'export_reporter_config',
    'export_algorithm_config',
]


def save_module(text, filename: Path):
    filename = filename.with_suffix('.py')
    if filename.exists():
        print(f"{colored('Skipped', 'yellow')} existing file {filename}")
        return
    with open(filename.with_suffix('.py'), 'w') as f:
        f.write(textwrap.dedent(text))
    print(f"{colored('Created', 'green')} {filename}")


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


def export_launch_module():
    """
    Export a default launch module template to 'run.py' file.

    The generated module includes basic launcher setup and execution code.
    """
    rows = """
    from pyfmto.experiments import Launcher


    if __name__ == '__main__':
        launcher = Launcher()
        launcher.run()\n
    """
    with open('run.py', 'w') as f:
        f.write(textwrap.dedent(rows))


def export_report_module():
    """
    Export a default report module template to 'report.py' file.

    The generated module includes basic reports setup with curve plotting enabled
    and other report formats commented out.
    """
    rows = """
    from pyfmto.experiments import Reports


    if __name__ == '__main__':
        reports = Reports()
        reports.to_curve()
        # reports.to_excel()
        # reports.to_violin()
        # reports.to_latex()\n
    """
    with open('report.py', 'w') as f:
        f.write(textwrap.dedent(rows))


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
        algs: tuple[str, ...] = ('BO', 'FMTBO'),
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


def export_default_config():
    """
    Export all default configurations.

    Exports launcher, reporter, algorithm and problem configurations
    with default settings in update mode.
    """
    export_launcher_config(mode='update')
    export_reporter_config(mode='update')
    export_algorithm_config(mode='update')
    export_problem_config(mode='update')


def export_demo(alg_name: str):
    """
    Export a complete demo setup for a new algorithm.

    Creates algorithm template, launch and report modules, and updates
    all configurations to include the new algorithm along with default 'BO'.

    Parameters
    ----------
    alg_name : str
        Name of the new algorithm to create demo for.
    """
    export_alg_template(alg_name)
    export_launch_module()
    export_report_module()
    export_launcher_config(algs=[alg_name, 'BO'], mode='update')
    export_reporter_config(algs=([alg_name, 'BO'], ), probs=('tetci2019_10d', 'arxiv2017_10d'), mode='update')
    export_algorithm_config(algs=(alg_name, 'BO'), mode='update')
    export_problem_config(mode='update')
