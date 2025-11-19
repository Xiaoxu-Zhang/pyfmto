from itertools import product
from pathlib import Path
from typing import Union, Any

from pyfmto.experiments.loaders import list_problems, list_algorithms, AlgorithmData, ProblemData
from pyfmto.utilities import parse_yaml, logger
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

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
    ):
        self.algorithm = algorithm
        self.problem = problem

    @property
    def root(self) -> Path:
        raise NotImplementedError

    def save_info(self):
        # save_yaml(self.root / "exp_conf.yaml")
        raise NotImplementedError

    def init_root(self):
        # self.root.mkdir(parents=True, exist_ok=True)
        pass


class ConfigParser:

    def __init__(self, config: dict):
        self.config_default = parse_yaml(CONF_DEFAULTS)
        self.config_update = config

    @property
    def config(self) -> dict:
        config = self.config_default.copy()
        for key, value in config.items():
            if key in self.config_update:
                config[key].update(value)
            else:
                config[key] = value
        return config

    @property
    def launcher(self) -> LauncherConfig:
        conf = LauncherConfig(**self.config['launcher'])
        algorithms = self.gen_alg_list()
        problems = self.gen_prob_list()
        conf.experiments = [ExperimentConfig(alg, prob) for alg, prob in product(algorithms, problems)]
        return conf

    @property
    def reporter(self) -> ReporterConfig:
        return ReporterConfig(**self.config['reporter'])

    @staticmethod
    def params_product(params: dict[str, Union[Any, list[Any]]]) -> list[dict[str, Any]]:
        values = []
        for key, value in params.items():
            if isinstance(value, str):
                values.append([value])
            else:
                values.append(value)
        result = []
        for combination in product(*values):
            result.append(dict(zip(params.keys(), combination)))
        return result

    def gen_alg_list(self) -> list[AlgorithmData]:
        algorithms: list[AlgorithmData] = []
        available_algs = list_algorithms()
        for alg_name in self.launcher.algorithms:
            alg_params = self.config.get('algorithms', {}).get(alg_name, {})
            real_name = alg_params.pop('base', '')
            name = alg_name if not real_name else real_name
            if name not in available_algs:
                logger.error(f"Algorithm {name} is not available.")
                continue
            alg_data = available_algs[name].copy()
            alg_data.set_name_alias(alg_name)
            alg_data.set_params_update(alg_params)
            algorithms.append(alg_data)
        return algorithms

    def gen_prob_list(self) -> list[ProblemData]:
        available_probs = list_problems()
        problems: list[ProblemData] = []
        for prob_name in self.launcher.problems:
            if prob_name not in available_probs:
                logger.error(f"Problem {prob_name} is not available.")
                continue
            prob_params = self.config.get('problems', {}).get(prob_name, {})
            params_variations = self.params_product(prob_params)
            for params in params_variations:
                prob_data = available_probs[prob_name].copy()
                prob_data.set_params_update(params)
                problems.append(prob_data)
        return problems
