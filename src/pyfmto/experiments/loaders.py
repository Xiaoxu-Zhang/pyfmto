import importlib
import inspect
import textwrap
import os
from importlib import import_module
from pathlib import Path
from typing import Type, Any

from pyfmto.problems import MultiTaskProblem
from pyfmto.framework import Client, Server
from pyfmto.problems import realworld, synthetic
from pyfmto.utilities import parse_yaml, logger, dumps_yaml


__all__ = [
    'list_problems',
    'load_problem',
    'init_problem',
    'list_algorithms',
    'load_algorithm',
]


class AlgorithmData:

    def __init__(self, name: str, client: Type[Client], server: Type[Server]):
        self.name: str = ''
        self.client: Type[Client] = client
        self.server: Type[Server] = server
        self.kwargs: dict[str, dict[str, Any]] = {}
        self.__get_kwargs()

    def __get_kwargs(self):
        c_doc = self.client.__doc__
        s_doc = self.server.__doc__
        c_args = parse_yaml(c_doc) if c_doc else {}
        s_args = parse_yaml(s_doc) if s_doc else {}
        if c_args:
            self.kwargs.update({'client': c_args})
        if s_args:
            self.kwargs.update({'server': s_args})

    @property
    def kwargs_yaml(self) -> str:
        if self.kwargs:
            return dumps_yaml(self.kwargs)
        else:
            return f"Algorithm '{self.name}' no configurable parameters."


class ProblemData:

    def __init__(self, problem: Type[MultiTaskProblem]):
        self.problem: Type[MultiTaskProblem] = problem
        self.kwargs: dict[str, Any] = {}
        self.__get_kwargs()

    @property
    def name(self) -> str:
        return self.problem.__name__

    def __get_kwargs(self):
        p_doc = self.problem.__doc__
        self.kwargs = parse_yaml(p_doc) if p_doc else {}

    @property
    def kwargs_yaml(self) -> str:
        if self.kwargs:
            return dumps_yaml(self.kwargs)
        else:
            return f"Problem '{self.name}' no configurable parameters."


def load_algorithm(name: str) -> AlgorithmData:
    name = name.upper()
    alg_dir = Path().cwd() / 'algorithms' / name
    if alg_dir.exists():
        module = importlib.import_module(f"algorithms.{name}")
    else:
        raise ModuleNotFoundError(f"Algorithm {name} not found.")

    clt, srv = None, None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if inspect.isclass(attr):
            if issubclass(attr, Client):
                clt = attr
            if issubclass(attr, Server):
                srv = attr
        if clt and srv:
            return AlgorithmData(name, clt, srv)

    msg: list[str] = [f'Load algorithm {name} failed:']
    if not clt:
        msg.append("  Client not found.")
    if not srv:
        msg.append("  Server not found.")
    raise ModuleNotFoundError('\n'.join(msg))


def list_algorithms(print_it=False):
    alg_dir = Path().cwd() / 'algorithms'
    if alg_dir.exists():
        folders = os.listdir(alg_dir)
        alg_names = [alg_name for alg_name in folders if alg_name.isupper()]
    else:
        alg_names = []

    algorithms: dict[str, AlgorithmData] = {}
    for name in alg_names:
        try:
            algorithms[name] = load_algorithm(name)
        except Exception as e:
            logger.error(f"Faild to load {name}: {e}")

    if print_it:
        if alg_dir.exists():
            alg_str = '\n'.join(list(algorithms.keys()))
            print(f"Found {len(algorithms)} available algorithms: \n{textwrap.indent(alg_str, ' ' * 4)}")
        else:
            print(f"'algorithms' folder not found in {alg_dir.parent}.")
    return algorithms


def init_problem(name: str, **kwargs) -> MultiTaskProblem:
    return load_problem(name).problem(**kwargs)


def load_problem(name: str) -> ProblemData:
    problems = list_problems()
    lowercase_map = {n.lower(): n for n in problems.keys()}
    if name.lower() in lowercase_map:
        return problems[lowercase_map[name.lower()]]
    else:
        raise ValueError(f"Problem '{name}' not found, call list_problems() to see available problems.")


def list_problems(print_it=False) -> dict[str, ProblemData]:
    problems: dict[str, ProblemData] = {}
    for module in [realworld, synthetic]:
        problems.update(collect_problems(module))
    prob_dir = Path().cwd() / 'problems'
    if prob_dir.exists():
        problems.update(collect_problems(import_module('problems')))
    if print_it:
        print("Available problems:")
        print(textwrap.indent('\n'.join(list(problems.keys())), ' ' * 4))
    return problems


def collect_problems(module) -> dict[str, ProblemData]:
    problems = {}
    for name in dir(module):
        cls = getattr(module, name)
        if inspect.isclass(cls) and issubclass(cls, MultiTaskProblem) and cls != MultiTaskProblem:
            problems[name] = ProblemData(cls)
    return problems
