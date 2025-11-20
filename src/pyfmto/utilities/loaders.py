import copy
import importlib
import inspect
import os
import textwrap
from importlib import import_module
from itertools import product
from pathlib import Path
from pydantic import BaseModel, field_validator, ConfigDict
from ruamel.yaml import CommentedMap
from typing import Type, Any, Union, Optional

from pyfmto.framework.client import Client
from pyfmto.framework.server import Server
from pyfmto.problems import MultiTaskProblem, realworld, synthetic
from .io import parse_yaml, dumps_yaml, load_yaml, save_yaml
from .loggers import logger
from .tools import show_in_table

__all__ = [
    'list_algorithms',
    'load_algorithm',
    'list_problems',
    'load_problem',
    'init_problem',
    'ProblemData',
    'ConfigParser',
    'AlgorithmData',
    'ExperimentConfig',
    'ReporterConfig',
    'LauncherConfig',
]


def recursive_to_pure_dict(data: Union[dict, CommentedMap]) -> dict[str, Any]:
    if not isinstance(data, (dict, CommentedMap)):
        return data
    for k, v in data.items():
        if isinstance(v, (dict, CommentedMap)):
            data[k] = recursive_to_pure_dict(dict(v))
    return data


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


def load_algorithm(name: str) -> 'AlgorithmData':
    name = name.upper()
    alg_dir = Path().cwd() / 'algorithms' / name
    if alg_dir.exists():
        module = importlib.import_module(f"algorithms.{name}")
    else:
        if alg_dir.parent.exists():
            raise ValueError(f"Algorithm {name} not found.")
        else:
            raise FileNotFoundError(f"'algorithms' folder not found in {alg_dir.parent.parent}.")

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


def init_problem(name: str, **kwargs) -> MultiTaskProblem:
    prob = load_problem(name)
    prob.set_params_update(kwargs)
    return prob.initialize()


def load_problem(name: str) -> 'ProblemData':
    problems = list_problems()
    lowercase_map = {n.lower(): n for n in problems.keys()}
    if name.lower() in lowercase_map:
        return problems[lowercase_map[name.lower()]]
    else:
        raise ValueError(f"Problem '{name}' not found, use 'pyfmto show problems' to see available problems.")


def list_problems(print_it=False) -> dict[str, 'ProblemData']:
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


def collect_problems(module) -> dict[str, 'ProblemData']:
    problems = {}
    for name in dir(module):
        cls = getattr(module, name)
        if inspect.isclass(cls) and issubclass(cls, MultiTaskProblem) and cls != MultiTaskProblem:
            problems[name] = ProblemData(cls)
    return problems


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

    def result_name(self, file_id: int):
        fe_init = self.problem.params.get('fe_init')
        fe_max = self.problem.params.get('fe_max')
        seed = self.problem.params.get('seed')
        fe_i = "" if fe_init is None else f"FEi{fe_init}_"
        fe_m = "" if fe_max is None else f"FEm{fe_max}_"
        seed = "" if seed is None else f"Seed{seed}_"
        filename = f"{fe_i}{fe_m}{seed}Rep{file_id:02d}.msgpack"
        return self.root / filename

    @property
    def num_results(self) -> int:
        prefix = self.result_name(0).name.split('Rep')[0]
        suffix = '.msgpack'
        results = [f for f in os.listdir(self.root) if f.startswith(prefix) and f.endswith(suffix)]
        return len(results)

    @property
    def params_dict(self) -> dict[str, Any]:
        data = {
            'algorithm': {
                'name': self.algorithm.name,
                'params': self.algorithm.params,
                'default': self.algorithm.params_default,
                'update': self.algorithm.params_update,
            },
            'problem': {
                'name': self.problem.name,
                'params': self.problem.params,
                'default': self.problem.params_default,
                'update': self.problem.params_update,
            }
        }
        return recursive_to_pure_dict(data)

    def backup_params(self):
        save_yaml(self.params_dict, self.root.parent / "exp_conf.yaml")

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


class LauncherConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: str = 'out/results'
    repeat: int = 1
    seed: int = 42
    backup: bool = True
    save: bool = True
    loglevel: str = 'INFO'
    algorithms: list[str]
    problems: list[str]
    experiments: list[ExperimentConfig] = []

    @field_validator('results', mode='before')
    def results_must_be_not_none(cls, v):
        if not isinstance(v, (str, type(None))):
            raise TypeError(f'results must be a string or None, got {type(v)} instead')
        return v if v is not None else 'out/results'

    @field_validator('loglevel')
    def loglevel_must_be_valid(cls, v):
        valid_values = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_values:
            raise ValueError(f'loglevel must be one of {valid_values}, got {v} instead')
        return v

    @field_validator('repeat', 'seed')
    def integer_must_be_positive(cls, v):
        if v < 1:
            raise ValueError('repeat must be >= 1')
        return v

    @field_validator('algorithms', 'problems')
    def lists_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('list must have at least 1 element')
        return v

    def show_summary(self):
        colored_tab, _ = show_in_table(
            num_exp=len(self.experiments),
            repeat_per_exp=self.repeat,
            total_repeat=self.total_repeat,
        )
        print(colored_tab)

    @property
    def n_exp(self) -> int:
        return len(self.experiments)

    @property
    def total_repeat(self) -> int:
        return self.n_exp * self.repeat


class ReporterConfig(BaseModel):
    results: Optional[str] = 'out/results'
    algorithms: list[list[str]]
    problems: list[str]

    @field_validator('results')
    def results_must_be_not_none(cls, v):
        return v if v is not None else 'out/results'

    @field_validator('algorithms')
    def inner_lists_must_have_min_length(cls, v):
        for inner_list in v:
            if len(inner_list) < 1:
                raise ValueError('inner lists must have at least 1 elements')
        return v

    @field_validator('problems', 'algorithms')
    def outer_list_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('problems list must have at least 1 element')
        return v


class ConfigParser:
    """
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

    def __init__(self, config: str = 'config.yaml'):
        self.config_default = parse_yaml(self.__class__.__doc__)
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
