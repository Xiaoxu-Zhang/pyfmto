from pathlib import Path
from typing import Union, Literal

from .client import Client, record_runtime
from .server import Server
from .packages import ClientPackage, SyncDataManager, DataArchive
from pyfmto.problems import load_problem
from pyfmto.utilities import save_yaml, load_yaml
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

__all__ = [
    'Client',
    'Server',
    'DataArchive',
    'SyncDataManager',
    'ClientPackage',
    'record_runtime',
    'export_problem_config',
    'export_launcher_config',
    'export_reporter_config',
    'export_algorithm_config',
]


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
            from pyfmto.experiments.utils import get_alg_kwargs
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
