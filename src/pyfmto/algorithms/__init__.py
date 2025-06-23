import os
import textwrap
from pathlib import Path
import importlib
import inspect
from typing import Any, Union
from ruamel.yaml import YAML
from yaml import MarkedYAMLError

from pyfmto.utilities import colored


def is_alg_name(name):
    return name.isupper()


def list_algorithms(print_it=False):
    algor_yours = os.listdir('algorithms') if os.path.exists('algorithms') else []
    algor_builtins = os.listdir(Path(__file__).parent)
    res = {
        'yours': [alg_name for alg_name in algor_yours if is_alg_name(alg_name)],
        'builtins': [alg_name for alg_name in algor_builtins if is_alg_name(alg_name)]
    }
    if print_it:
        print(colored("Yours:", 'yellow'))
        alg_str = '\n'.join(res['yours'])
        print(textwrap.indent(alg_str, ' ' * 2))
        alg_str = '\n'.join(res['builtins'])
        print(colored('Builtins:', 'blue'))
        print(textwrap.indent(alg_str, ' ' * 2))
    return res


def load_algorithm(name):
    all_alg = list_algorithms()
    if name in all_alg['yours']:
        module = importlib.import_module(f"algorithms.{name}")
        is_builtin_alg = False
    elif name in all_alg['builtins']:
        module = importlib.import_module(f"pyfmto.algorithms.{name}")
        is_builtin_alg = True
    else:
        raise RuntimeError(f'algorithm {name} not found.')
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

def export_kwargs(algorithms: list[str], directory: str=None):
    data = {name: get_alg_kwargs(name) for name in algorithms}
    fdir = Path(directory) if directory is not None else Path.cwd()
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(fdir / f"default_kwargs.yaml", 'w') as f:
        yaml.dump({'algorithms': data}, f)

def get_alg_kwargs(name: str):
    docstr = _collect_docstr(name)
    cleaned_lines = [line for line in docstr.splitlines() if not line.strip() == '']
    cleaned_text = "\n".join(cleaned_lines)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    try:
        data = yaml.load(cleaned_text)
    except MarkedYAMLError:
        raise
    return data

def _collect_docstr(name):
    try:
        cls = load_algorithm(name)
    except RuntimeError as e:
        print(e)
        return ''

    clt = cls['client'].__doc__
    srv = cls['server'].__doc__
    if not clt and not srv:
        return "{}"
    res = ''
    if srv:
        res += f'    server:\n{textwrap.indent(srv, " " * 4)}\n'
    if clt:
        res += f'    client:\n{textwrap.indent(clt, " " * 4)}\n'
    return res