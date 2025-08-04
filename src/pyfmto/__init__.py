__version__ = "0.0.1"

from pathlib import Path
from typing import Union, Literal

from pyfmto.algorithms import load_algorithm, get_alg_kwargs
from pyfmto.problems import load_problem
from pyfmto.utilities import save_yaml, load_yaml, parse_yaml
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

__all__ = [
    'load_problem',
    'load_algorithm',
    'export_launcher_config',
    'export_reporter_config',
    'export_algorithm_config',
    'export_problem_config',
    'export_default_config',
]


def export_launcher_config(
        results: str='out/results',
        repeat: int=1,
        save: bool=True,
        seed: int=42,
        backup: bool=True,
        algs: Union[tuple[str], list[str]]=('BO', 'FMTBO'),
        probs: Union[tuple[str], list[str]]=('tetci2019', 'arxiv2017'),
        mode: Literal['new', 'update']='new',
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
        results: str='out/results',
        algs: list[list[str]]=(('BO', 'FMTBO'), ),
        probs: list[str]=('tetci2019', 'arxiv2017'),
        mode: Literal['new', 'update']='new',
):
    data = ReporterConfig(
        results=results,
        algorithms=algs,
        problems=probs,
    )
    return save_config('reporter', data.model_dump(), mode)


def export_algorithm_config(
        algs: list[str]=('BO', 'FMTBO'),
        mode: Literal['new', 'update']= 'new'
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
        probs: list[str]=('arxiv2017', 'tetci2019'),
        mode: Literal['new', 'update']='new'
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


def save_config(name: str, data, mode: Literal['new', 'update']='new'):
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
