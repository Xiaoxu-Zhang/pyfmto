import copy
import importlib
import inspect
import os
import textwrap
from deepdiff import DeepDiff
from importlib import import_module
from itertools import product
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from ruamel.yaml import CommentedMap
from textwrap import indent
from typing import Type, Any, Union, Literal

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
    'ConfigLoader',
    'AlgorithmData',
    'ExperimentConfig',
    'ReporterConfig',
    'LauncherConfig',
]


def recursive_to_pure_dict(data: Any) -> dict[str, Any]:
    """
    Recursively convert nested dict and CommentedMap objects to a pure Python
    dictionary to avoid YAML serialization issues.
    """
    if isinstance(data, (dict, CommentedMap)):
        for k, v in data.items():
            data[k] = recursive_to_pure_dict(v)
    else:
        return data
    return data


def combine_params(params: dict[str, Union[Any, list[Any]]]) -> list[dict[str, Any]]:
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


def list_algorithms(print_it=False):
    alg_dir = Path().cwd() / 'algorithms'
    if alg_dir.exists():
        folders = os.listdir(alg_dir)
        alg_names = [alg_name for alg_name in folders if alg_name.isupper()]
    else:
        logger.error(f"'algorithms' folder not found in {alg_dir.parent}.")
        alg_names = []

    algorithms: dict[str, AlgorithmData] = {}
    for name in alg_names:
        try:
            algorithms[name] = load_algorithm(name)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

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
    prob.params_update = kwargs
    return prob.initialize()


def load_problem(name: str) -> 'ProblemData':
    problems = list_problems()
    if name in problems.keys():
        return problems[name]
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
        print(f"Found {len(problems)} available problems:")
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

    def __init__(self, name: str, client: Any = None, server: Any = None):
        self.name_orig = name
        self.name_alias = ''
        self.client: Type[Client] = client
        self.server: Type[Server] = server
        self.params_default: dict[str, dict[str, Any]] = {}
        self.params_update: dict[str, dict[str, Any]] = {}
        self.__parse_default_params()

    def __parse_default_params(self):
        c_doc = self.client.__doc__ if self.client else None
        s_doc = self.server.__doc__ if self.server else None
        c_args = parse_yaml(c_doc) if c_doc else {}
        s_args = parse_yaml(s_doc) if s_doc else {}
        if c_args:
            self.params_default.update({'client': c_args})
        if s_args:
            self.params_default.update({'server': s_args})

    def copy(self) -> 'AlgorithmData':
        return copy.deepcopy(self)

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        kwargs = copy.deepcopy(self.params_default)
        for k in ['client', 'server']:
            for k2, v2 in self.params_update.get(k, {}).items():
                if k not in kwargs:
                    kwargs[k] = {}
                kwargs[k][k2] = v2
        return kwargs

    @property
    def params_diff(self) -> str:
        return DeepDiff(self.params_default, self.params).pretty()

    @property
    def name(self) -> str:
        return self.name_orig if not self.name_alias else self.name_alias

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
            'npd': 1,
            'random_ctrl': 'weak',
            'seed': 123,
        }
        self.params_update: dict[str, Any] = {}
        self.__parse_default_params()

    def __parse_default_params(self):
        p_doc = self.problem.__doc__
        self.params_default.update(parse_yaml(p_doc))

    def copy(self) -> 'ProblemData':
        return copy.deepcopy(self)

    @property
    def params(self) -> dict[str, Any]:
        params = copy.deepcopy(self.params_default)
        params.update(self.params_update)
        dim = params.get('dim', 0)
        if dim > 0:
            if 'fe_init' not in params:
                params.update(fe_init=5 * dim)
            if 'fe_max' not in params:
                params.update(fe_max=11 * dim)
        return params

    @property
    def params_diff(self) -> str:
        return DeepDiff(self.params_default, self.params).pretty()

    def initialize(self) -> MultiTaskProblem:
        return self.problem(**self.params)

    @property
    def name(self) -> str:
        base_name = self.problem.__name__
        return f"{base_name}_{self.dim_str}" if self.dim > 0 else base_name

    @property
    def npd(self) -> int:
        return self.params.get('npd', 0)

    @property
    def npd_str(self) -> str:
        return f"NPD{self.npd}" if self.npd > 0 else ""

    @property
    def dim(self) -> int:
        return self.params.get('dim', 0)

    @property
    def dim_str(self) -> str:
        if self.dim > 0:
            return f"{self.dim}D"
        else:
            return ""

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
        return self._root / self.algorithm.name / self.problem.name / self.problem.npd_str

    @property
    def prefix(self) -> str:
        fe_init = self.problem.params.get('fe_init')
        fe_max = self.problem.params.get('fe_max')
        seed = self.problem.params.get('seed')
        fe_i = "" if fe_init is None else f"FEi{fe_init}_"
        fe_m = "" if fe_max is None else f"FEm{fe_max}_"
        seed = "" if seed is None else f"Seed{seed}_"
        return f"{fe_i}{fe_m}{seed}"

    def result_name(self, file_id: int):
        return self.root / f"{self.prefix}Rep{file_id:02d}.msgpack"

    @property
    def num_results(self) -> int:
        if not self.root.exists():
            return 0
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
        self.root.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(self.params_dict, self.root.parent / "parameters.yaml")

    def init_root(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        info = [
            f"alg: {self.algorithm.name}"
            f"prob: {self.problem.name}"
            f"  prob.npd: {self.problem.params['npd']}"
            f"  prob.dim: {self.problem.params.get('dim', 'unknown')}"
        ]
        return '\n'.join(info)


class LauncherConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: str
    repeat: int
    seed: int
    backup: bool
    save: bool
    loglevel: str
    algorithms: list[str]
    problems: list[str]
    experiments: list[ExperimentConfig] = []

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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: str
    algorithms: list[list[str]]
    problems: list[str]
    formats: list[str]
    params: dict[str, Any] = {}
    experiments: list[ExperimentConfig] = []
    groups: list[tuple[list[str], str, str]] = []

    @property
    def root(self) -> Path:
        return Path(self.results)


class ConfigLoader:
    """
    launcher:
        results: out/results  # [optional] save results to this directory
        repeat: 2             # [optional] repeat each experiment for this number of times
        seed: 123             # [optional] random seed
        save: true            # [optional] save results to disk
        backup: false          # [optional] backup parameters to disk
        loglevel: INFO        # [optional] log level [CRITICAL, ERROR, WARNING, INFO, DEBUG], default INFO
        algorithms: []        # run these algorithms
        problems: []          # run each algorithm on these problems
    reporter:
        formats: [excel]      # [optional] generate these reports
    """

    def __init__(self, config: str = 'config.yaml'):
        self.config_default = parse_yaml(self.__class__.__doc__)
        self.config_update = load_yaml(config)
        self.config = copy.deepcopy(self.config_default)

        self.merge_global_config_from_updates()
        self.fill_launcher_config_from_reporter()

    def merge_global_config_from_updates(self):
        for key, value in self.config_update.items():
            if key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value

    def fill_launcher_config_from_reporter(self) -> None:
        launcher_params = self.config['launcher']
        for key in ['results', 'problems', 'algorithms']:
            if key in self.config['reporter']:
                continue
            if key == 'algorithms':
                self.config['reporter'][key] = [launcher_params[key]]
            else:
                self.config['reporter'][key] = launcher_params[key]

    @property
    def launcher(self) -> LauncherConfig:
        self.check_config_issues('launcher')
        conf = LauncherConfig(**self.config['launcher'])
        algorithms = self.gen_alg_list(conf.algorithms)
        problems = self.gen_prob_list(conf.problems)
        logger.debug(f"algorithms: {[alg.name for alg in algorithms]}")
        logger.debug(f"problems: {[prob.name for prob in problems]}")
        conf.experiments = [ExperimentConfig(alg, prob, conf.results) for alg, prob in product(algorithms, problems)]
        return conf

    @property
    def reporter(self) -> ReporterConfig:
        self.check_config_issues('reporter')
        conf = ReporterConfig(**self.config['reporter'])
        alg_names = list(set(sum(conf.algorithms, [])))
        algorithms = [AlgorithmData(name) for name in alg_names]
        problems = self.gen_prob_list(conf.problems)
        conf.experiments = [
            ExperimentConfig(alg, prob, conf.results)
            for alg, prob in product(algorithms, problems)
        ]
        conf.groups = [
            (algs, prob.name, prob.npd_str)
            for algs, prob in list(product(conf.algorithms, problems))
        ]
        return conf

    def gen_alg_list(self, names: list[str]) -> list[AlgorithmData]:
        algorithms: list[AlgorithmData] = []
        available_algs = list_algorithms()
        for name_alias in names:
            alg_params = self.config.get('algorithms', {}).get(name_alias, {})
            alg_name = alg_params.pop('base', name_alias)
            if alg_name not in available_algs:
                logger.error(f"Algorithm {alg_name} is not available.")
                continue
            alg_data = available_algs[alg_name].copy()
            alg_data.name_alias = name_alias
            alg_data.params_update = alg_params
            algorithms.append(alg_data)
        return algorithms

    def gen_prob_list(self, names: list[str]) -> list[ProblemData]:
        available_probs = list_problems()
        problems: list[ProblemData] = []
        for n in names:
            if n not in available_probs:
                logger.error(f"Problem {n} is not available.")
                continue
            prob_params = self.config.get('problems', {}).get(n, {})
            params_variations = combine_params(prob_params)
            for params in params_variations:
                prob_data = available_probs[n].copy()
                prob_data.params_update = params
                problems.append(prob_data)
        return problems

    def check_config_issues(self, name: Literal['launcher', 'reporter']) -> None:
        logger.debug(f"Checking {name} configuration...")
        if name == 'launcher':
            issues = self.check_launcher_config()
        else:
            issues = self.check_reporter_config()
        if issues:
            detail = indent('\n'.join(issues), ' ' * 4)
            msg = f"{name.title()} configuration issues:\n{detail}"
            logger.error(msg)
            raise ValueError(msg)

    def check_launcher_config(self) -> list[str]:
        issues = []
        launcher = self.config['launcher']
        loglevel = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        if not launcher.get('results'):
            issues.append("No results directory specified in launcher.")
        if launcher.get('repeat') <= 0:
            issues.append("Invalid repeat number specified in launcher. Must be greater than 0.")
        if not isinstance(launcher.get('save'), bool):
            issues.append("Invalid save option specified in launcher. Must be True or False.")
        if launcher.get('loglevel') not in loglevel:
            issues.append(f"Invalid log level specified in launcher. Choices: {loglevel}")
        if not launcher.get('algorithms'):
            issues.append("No algorithms specified in launcher.")
        if not launcher.get('problems'):
            issues.append("No problems specified in launcher.")
        return issues

    def check_reporter_config(self) -> list[str]:
        issues = []
        reporter = self.config['reporter']
        if not reporter.get('results'):
            issues.append("No results directory specified in reporter or launcher.")
        if not reporter.get('algorithms', []):
            issues.append("No algorithms specified in reporter or launcher.")
        else:
            validate_values: list[list[str]] = []
            for item in reporter['algorithms']:
                if isinstance(item, str) and item:
                    validate_values.append([item])
                elif isinstance(item, list) and item:
                    validate_values.append(item)
                else:
                    issues.append(f"Invalid value [type:{type(item)}, value:{item}] specified in reporter.")
                reporter['algorithms'] = validate_values
        return issues
