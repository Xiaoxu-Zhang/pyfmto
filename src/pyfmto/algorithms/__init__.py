import os
import textwrap
import importlib
import inspect
from pathlib import Path

from pyfmto.framework import Client, Server
from pyfmto.utilities import parse_yaml, colored

__all__ = ['list_algorithms', 'load_algorithm', 'get_alg_kwargs']


class Algorithm:
    name: str
    is_builtin: bool
    client: Client
    server: Server

    @property
    def is_complete(self) -> bool:
        return hasattr(self, 'client') and hasattr(self, 'server')


def load_algorithm(name: str):
    alg = Algorithm()
    alg.name = name
    all_alg = list_algorithms()
    if name in all_alg['yours']:
        module = importlib.import_module(f"algorithms.{name}")
        alg.is_builtin = False
    elif name in all_alg['builtins']:
        module = importlib.import_module(f"pyfmto.algorithms.{name}")
        alg.is_builtin = True
    else:
        raise ValueError(f'algorithm {name} not found.')

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if inspect.isclass(attr):
            if issubclass(attr, Client):
                alg.client = attr
            if issubclass(attr, Server):
                alg.server = attr
        if alg.is_complete:
            break

    msg: list[str] = [f'load algorithm {name} failed:']
    if not hasattr(alg, 'client'):
        msg.append("  Client not found.")
    if not hasattr(alg, 'server'):
        msg.append("  Server not found.")
    if len(msg) > 1:
        raise RuntimeError('\n'.join(msg))
    return alg


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
    clt = alg_data.client.__doc__
    srv = alg_data.server.__doc__
    data = {}
    if clt:
        data.update({'client': parse_yaml(clt)})
    if srv:
        data.update({'server': parse_yaml(srv)})
    return data
