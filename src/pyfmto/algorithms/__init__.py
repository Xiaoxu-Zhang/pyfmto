import os
import textwrap
import importlib
import inspect
from pathlib import Path
from typing import Any
from pyfmto.utilities import parse_yaml, colored

__all__ = ['list_algorithms', 'load_algorithm', 'get_alg_kwargs']


def load_algorithm(name: str):
    all_alg = list_algorithms()
    if name in all_alg['yours']:
        module = importlib.import_module(f"algorithms.{name}")
        is_builtin_alg = False
    elif name in all_alg['builtins']:
        module = importlib.import_module(f"pyfmto.algorithms.{name}")
        is_builtin_alg = True
    else:
        raise ValueError(f'algorithm {name} not found.')
    attr_names = dir(module)
    res: dict[str: Any] = {}
    for attr_name in attr_names:
        attr = getattr(module, attr_name)
        if inspect.isclass(attr):
            if 'Client' in attr_name:
                res['client'] = attr
            if 'Server' in attr_name:
                res['server'] = attr
        if len(res) == 2:
            break
    if len(res) != 2:
        raise RuntimeError(f'load algorithm {name} failed, load result is {res}')
    res.update(is_builtin_alg=is_builtin_alg)
    res.update(name=name)
    return res


def list_algorithms(print_it=False):
    algor_yours = os.listdir('algorithms') if os.path.exists('algorithms') else []
    algor_builtins = os.listdir(Path(__file__).parent)
    res = {
        # We recognize the algorithm name based on whether the name is in uppercase.
        'yours': [alg_name for alg_name in algor_yours if alg_name.isupper()],
        'builtins': [alg_name for alg_name in algor_builtins if alg_name.isupper()]
    }
    if print_it:
        print(colored("Yours:", 'yellow'))
        alg_str = '\n'.join(res['yours'])
        print(textwrap.indent(alg_str, ' ' * 2))
        alg_str = '\n'.join(res['builtins'])
        print(colored('Builtins:', 'blue'))
        print(textwrap.indent(alg_str, ' ' * 2))
    return res


def get_alg_kwargs(name: str):
    alg_data = load_algorithm(name)
    clt = alg_data['client'].__doc__
    srv = alg_data['server'].__doc__
    data = {}
    if clt:
        data.update({'client': parse_yaml(clt)})
    if srv:
        data.update({'server': parse_yaml(srv)})
    return data