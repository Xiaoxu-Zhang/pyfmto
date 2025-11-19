import copy
import importlib
import inspect
import os
import textwrap
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Type, Any, Union

from ruamel.yaml import CommentedMap

from pyfmto.framework.client import Client
from pyfmto.framework.server import Server
from pyfmto.problems import MultiTaskProblem, realworld, synthetic
from .io import parse_yaml, dumps_yaml, load_yaml, save_yaml
from .loggers import logger
from .schemas import LauncherConfig, ReporterConfig


class AlgorithmData:

    def __init__(self, name: str, client: Type[Client], server: Type[Server]):
        self.name_orig = name
        self.name_alias = ''
        self.client: Type[Client] = client
        self.server: Type[Server] = server
        self.params_default: dict[str, dict[str, Any]] = {}
        self.params_update: dict[str, dict[str, Any]] = {}
        self.__parse_default_params()

    def __parse_default_params(self):
        c_doc = self.client.__doc__
        s_doc = self.server.__doc__
        c_args = parse_yaml(c_doc) if c_doc else {}
        s_args = parse_yaml(s_doc) if s_doc else {}
        if c_args:
            self.params_default.update({'client': c_args})
        if s_args:
            self.params_default.update({'server': s_args})

    def set_params_update(self, params_update: dict[str, dict[str, Any]]):
        self.params_update.update(params_update)

    def set_name_alias(self, alias: str):
        self.name_alias = alias

    def copy(self) -> 'AlgorithmData':
        return copy.deepcopy(self)

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        kwargs = self.params_default.copy()
        for k, v in self.params_update.items():
            if k in kwargs:
                kwargs[k].update(v)
            else:
                kwargs.update({k: v})
        return kwargs

    def params_diff(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.name_orig if self.name_alias == '' else self.name_alias

    @property
    def params_yaml(self) -> str:
        if self.params_default:
            return dumps_yaml(self.params_default)
        else:
            return f"Algorithm '{self.name}' no configurable parameters."


class ProblemData:

    def __init__(self, problem: Type[MultiTaskProblem]):
        self.problem: Type[MultiTaskProblem] = problem
        self.params_default: dict[str, Any] = {
            'np_per_dim': 1,
            'random_ctrl': 'weak',
            'seed': 123,
        }
        self.params_update: dict[str, Any] = {}
        self.__parse_default_params()

    def __parse_default_params(self):
        p_doc = self.problem.__doc__
        self.params_default.update(parse_yaml(p_doc))
        if 'dim' in self.params_default:
            dim = self.params_default['dim']
            self.params_default.update(
                fe_init=5 * dim,
                fe_max=11 * dim,
            )

    def copy(self) -> 'ProblemData':
        return copy.deepcopy(self)

    def set_params_update(self, params_update: dict[str, dict[str, Any]]):
        self.params_update.update(params_update)

    @property
    def params(self) -> dict[str, Any]:
        params = self.params_default.copy()
        params.update(self.params_update)
        return params

    def initialize(self) -> MultiTaskProblem:
        return self.problem(**self.params)

    @property
    def name(self) -> str:
        return self.problem.__name__

    @property
    def params_yaml(self) -> str:
        if self.params_default:
            return dumps_yaml(self.params_default)
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
        raise ValueError(f"Problem '{name}' not found, use 'pyfmto show problems' to see available problems.")


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


CONF_DEFAULTS = """
launcher:
    results: out/results  # [optional] save results to this directory
    repeat: 2             # [optional] repeat each experiment for this number of times
    save: true            # [optional] save results to disk
    loglevel: INFO        # [optional] log level [CRITICAL, ERROR, WARNING, INFO, DEBUG], default INFO
    algorithms: []        # run these algorithms
    problems: []          # run each algorithm on these problems
reporter:
    results: out/results  # [optional] load results from this directory
    formats: [excel]      # [optional] generate these reports
    algorithms: []        # make comparison on these groups of algorithms
    problems: []          # use results that algorithms runs on these problems
"""


class ExperimentConfig:

    def __init__(
            self,
            algorithm: AlgorithmData,
            problem: ProblemData,
            root: str
    ):
        self.algorithm = algorithm
        self.problem = problem
        self._root = Path(root)

    @property
    def root(self) -> Path:
        alg_name = self.algorithm.name
        prob_name = self.problem.name
        if 'dim' in self.problem.params:
            prob_name = f"{prob_name}_{self.problem.params['dim']}D"
        npd = self.problem.params['np_per_dim']
        return self._root / alg_name / prob_name / f"NPD{npd}"

    def save_info(self):
        data = {
                'alg_params': self.algorithm.params,
                'alg_default': self.algorithm.params_default,
                'alg_update': self.algorithm.params_update,
                'prob_params': self.problem.params,
                'prob_default': self.problem.params_default,
                'prob_update': self.problem.params_update,
            }
        save_yaml(
            CommentedMap(data),
            self.root.parent / "exp_conf.yaml")
        return data

    def init_root(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        info = [
            f"alg: {self.algorithm.name}"
            f"prob: {self.problem.name}"
            f"  prob.npd: {self.problem.params['np_per_dim']}"
            f"  prob.dim: {self.problem.params.get('dim', 'unknown')}"
        ]
        return '\n'.join(info)


class ConfigParser:

    def __init__(self, config: str = 'config.yaml'):
        self.config_default = parse_yaml(CONF_DEFAULTS)
        self.config_update = load_yaml(config)

    @property
    def config(self) -> dict:
        config = self.config_default.copy()
        for key, value in self.config_update.items():
            if key in config:
                config[key].update(value)
            else:
                config[key] = value
        return config

    @property
    def launcher(self) -> LauncherConfig:
        conf = LauncherConfig(**self.config['launcher'])
        algorithms = self.gen_alg_list()
        problems = self.gen_prob_list()
        conf.experiments = [ExperimentConfig(alg, prob, conf.results) for alg, prob in product(algorithms, problems)]
        return conf

    @property
    def reporter(self) -> ReporterConfig:
        return ReporterConfig(**self.config['reporter'])

    @staticmethod
    def params_product(params: dict[str, Union[Any, list[Any]]]) -> list[dict[str, Any]]:
        values = []
        for key, value in params.items():
            if isinstance(value, list):
                values.append(value)
            else:
                values.append([value])
        result = []
        for combination in product(*values):
            result.append(dict(zip(params.keys(), combination)))
        return result

    def gen_alg_list(self) -> list[AlgorithmData]:
        algorithms: list[AlgorithmData] = []
        available_algs = list_algorithms()
        for name_alias in self.config['launcher']['algorithms']:
            alg_params = self.config.get('algorithms', {}).get(name_alias, {})
            alg_name = alg_params.pop('base', name_alias)
            if alg_name not in available_algs:
                logger.error(f"Algorithm {alg_name} is not available.")
                continue
            alg_data = available_algs[alg_name].copy()
            alg_data.set_name_alias(name_alias)
            alg_data.set_params_update(alg_params)
            algorithms.append(alg_data)
        return algorithms

    def gen_prob_list(self) -> list[ProblemData]:
        available_probs = list_problems()
        lower_case_map = {n.lower(): n for n in available_probs.keys()}
        problems: list[ProblemData] = []
        for prob_name in self.config['launcher']['problems']:
            if prob_name not in lower_case_map:
                logger.error(f"Problem {prob_name} is not available.")
                continue
            prob_params = self.config.get('problems', {}).get(prob_name, {})
            params_variations = self.params_product(prob_params)
            for params in params_variations:
                prob_data = available_probs[lower_case_map[prob_name]].copy()
                prob_data.set_params_update(params)
                problems.append(prob_data)
        return problems
