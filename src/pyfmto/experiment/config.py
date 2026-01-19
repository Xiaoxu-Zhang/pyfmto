import copy
import inspect
import os
import shutil
import textwrap
from collections.abc import Iterable
from datetime import datetime
from itertools import chain, product
from pathlib import Path
from textwrap import indent
from typing import Any, Literal, Union, no_type_check

import psutil
from pydantic import BaseModel, ConfigDict
from rich import box
from rich.console import Console
from rich.table import Table

from ..core.typing import TComponentList, TComponentNames
from ..framework import AlgorithmData
from ..problem import ProblemData
from ..utilities.io import load_yaml, parse_yaml, recursive_to_pure_dict
from ..utilities.loaders import discover
from ..utilities.loggers import logger
from ..utilities.tools import clear_console, deepmerge, get_os_name, titled_tabulate

__all__ = [
    "ConfigLoader",
    "ExperimentData",
    "LauncherConfig",
    "ReporterConfig",
]


def combine_params(params: dict[str, Union[Any, list[Any]]]) -> list[dict[str, Any]]:
    values = []
    for v in params.values():
        if isinstance(v, list):
            values.append(v)
        else:
            values.append([v])
    result = []
    for combination in product(*values):
        result.append(dict(zip(params.keys(), combination)))
    return result


class ExperimentData:

    def __init__(
            self,
            algorithm: AlgorithmData,
            problem: ProblemData,
            results: str
    ):
        self.algorithm = algorithm
        self.problem = problem
        self._result_root = Path(results)

    @property
    def result_dir(self) -> Path:
        return self._result_root / self.algorithm.name / self.problem.name / self.problem.npd_str

    @property
    def snapshot_dir(self) -> Path:
        return self.result_dir.parent / f"snapshot_{datetime.now().strftime('%Y-%m-%d')}"

    @property
    def code_dest(self) -> Path:
        return self.snapshot_dir / "code"

    @property
    def markdown_dest(self) -> Path:
        return self.snapshot_dir / "environment.md"

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
        return self.result_dir / f"{self.prefix}Rep{file_id:02d}.msgpack"

    @property
    def n_results(self) -> int:
        if not self.result_dir.exists():
            return 0
        prefix = self.result_name(0).name.split('Rep')[0]
        suffix = '.msgpack'
        results = [f for f in os.listdir(self.result_dir) if f.startswith(prefix) and f.endswith(suffix)]
        return len(results)

    @property
    def params_snapshot(self) -> dict[str, Any]:
        data = {
            'algorithm': self.algorithm.params_snapshot,
            'problem': self.problem.params_snapshot,
        }
        return recursive_to_pure_dict(data)

    @staticmethod
    def desc_sys():
        from pyfmto.utilities.tools import get_cpu_model
        data = {
            'OS': get_os_name(),
            "CPU": get_cpu_model(),
            "MEM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 1)} GB"
        }
        return '\n'.join([f"- {k}: {v}" for k, v in data.items()])

    @staticmethod
    def desc_env(packages: list[str]):
        import platform

        from pyfmto.utilities.tools import get_pkgs_version
        data = {
            'python': platform.python_version(),
            **get_pkgs_version(packages)
        }
        return '\n'.join([f"- {k}: `{v}`" for k, v in data.items()])

    def create_snapshot(self, packages: list[str]):
        if self.snapshot_dir.exists():
            return
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%H:%M:%S")
        template = f"""
            # Experiment Information

            This snapshot was created on `{date}` at `{time}`

            ---

            ## system

            <system>

            ## environment

            <environment>

            ## Configuration

            ### Algorithm

            ``` yaml
            <algorithm>
            ```

            ### Problem

            ``` yaml
            <problem>
            ```
        """
        md_str = textwrap.dedent(template)
        md_str = md_str.replace('<system>', self.desc_sys())
        md_str = md_str.replace('<environment>', self.desc_env(packages))
        md_str = md_str.replace('<algorithm>', self.algorithm.params_snapshot)
        md_str = md_str.replace('<problem>', self.problem.params_snapshot)

        file_src = inspect.getfile(self.algorithm.client)
        code_dest = self.snapshot_dir / 'code'
        md_file = self.snapshot_dir / 'environment.md'
        for p in Path(file_src).resolve().parents:
            if p.parent.name == 'algorithms':
                shutil.copytree(p, code_dest)
        with open(md_file, 'w') as f:
            f.write(md_str)

    def init_root(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        tab = {
            'Algorithm': [self.algorithm.name],
            'Problem': [self.problem.name],
            'NPD': [self.problem.npd_str],
            'Dimension': [self.problem.dim_str],
        }
        return titled_tabulate("Experiment", '=', tab, tablefmt='rounded_grid')

    def __repr__(self):
        info = [
            f"Alg({self.algorithm.name})",
            f"Prob({self.problem.name})",
            f"NPD({self.problem.params['npd']})",
            f"Dim({self.problem.params.get('dim', '-')})",
        ]

        return ' '.join(info)


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: str
    algorithms: list[str] = []
    problems: list[str] = []
    algorithms_data: list[AlgorithmData] = []
    problems_data: list[ProblemData] = []

    @property
    def experiments(self) -> Iterable[ExperimentData]:
        return (
            ExperimentData(alg, prob, self.results)
            for alg, prob in product(self.algorithms_data, self.problems_data)
        )

    @property
    def n_exp(self) -> int:
        return len(self.algorithms_data) * len(self.problems_data)


class LauncherConfig(Config):
    sources: list[str]
    repeat: int
    seed: int
    save: bool
    loglevel: str
    snapshot: bool
    verbose: bool
    packages: list[str]

    def show_summary(self):
        tab = Table(
            title="Experiments Summary",
            title_justify="center",
            box=box.ROUNDED,
        )
        tab.add_column('Algorithm', justify='center', style="cyan")
        tab.add_column('Original', justify='center', style="cyan")
        tab.add_column('Problem', justify='center', style="magenta")
        tab.add_column('NPD', justify='center', style="yellow")
        tab.add_column('Success', justify='center')

        for exp in self.experiments:
            tab.add_row(
                exp.algorithm.name,
                exp.algorithm.name_orig,
                exp.problem.name,
                exp.problem.npd_str,
            )
        clear_console()
        Console().print(tab)

    @property
    def total_repeat(self) -> int:
        return self.n_exp * self.repeat


class ReporterConfig(Config):
    comparisons: list[list[str]]
    formats: list[str]
    params: dict[str, Any] = {}

    @property
    def groups(self) -> Iterable[tuple[list[str], str, str]]:
        return (
            (algs, prob.name, prob.npd_str)
            for algs, prob in list(product(self.comparisons, self.problems_data))
        )


class ConfigLoader:
    """
    launcher:
        sources: []           # load alg/prob in these directory
        results: out/results  # [optional] save results to this directory
        repeat: 2             # [optional] repeat each experiment for this number of times
        seed: 123             # [optional] random seed
        save: true            # [optional] save results to disk
        loglevel: INFO        # [optional] log level [CRITICAL, ERROR, WARNING, INFO, DEBUG], default INFO
        snapshot: true        # [optional] If create snapshot of the experiment
        verbose: false        # [optional] Save detailed information for each repeat run
        packages: []          # [optional] Record the version of these packages
        algorithms: []        # run these algorithms
        problems: []          # run each algorithm on these problems
    reporter:
        formats: [excel]      # [optional] generate these reports
    """

    def __init__(self, config: Union[str, Path] = 'config.yaml'):
        self.config_default = parse_yaml(self.__class__.__doc__)
        self.config_update = load_yaml(config, ignore_errors=True)
        self.config = copy.deepcopy(self.config_default)
        self.merge_global_config_from_updates()
        self.fill_reporter_config_from_launcher()
        logger.setLevel(self.config['launcher']['loglevel'])

    @property
    def sources(self) -> list[str]:
        return self.config['launcher']['sources']

    def merge_global_config_from_updates(self):
        deepmerge(self.config, self.config_update, lock=False)
        cwd = str(Path().cwd().resolve())
        if cwd not in self.config['launcher']['sources']:
            self.config['launcher']['sources'].append(cwd)

    def fill_reporter_config_from_launcher(self) -> None:
        launcher_params = self.config['launcher']
        for key in ['results', 'problems', 'comparisons']:
            if key in self.config['reporter']:
                continue
            if key == 'comparisons':
                self.config['reporter'][key] = [launcher_params['algorithms']]
            else:
                self.config['reporter'][key] = launcher_params[key]

    @property
    def launcher(self) -> LauncherConfig:
        self.check_config_issues('launcher')
        logger.setLevel(self.config['launcher']['loglevel'])
        return self.fill_components(LauncherConfig(**self.config['launcher']))

    @property
    def reporter(self) -> ReporterConfig:
        self.check_config_issues('reporter')
        conf = ReporterConfig(**self.config['reporter'])
        conf.algorithms = list(set(chain.from_iterable(conf.comparisons)))
        return self.fill_components(conf)

    @no_type_check
    def fill_components(self, conf: Union[LauncherConfig, ReporterConfig]) -> Union[LauncherConfig, ReporterConfig]:
        conf.algorithms_data = self.collect_components('algorithms', conf.algorithms)
        conf.problems_data = self.collect_components('problems', conf.problems)
        return conf

    def collect_components(self, target: TComponentNames, names: list[str]) -> TComponentList:
        components = discover(self.sources)
        res: TComponentList = []
        for name_alias in names:
            settings = self.config.get(target, {}).get(name_alias, {})
            name_orig = settings.pop('base', name_alias)
            logger.debug(f"Looking for {target.title()} '{name_orig}' settings={settings!r}")
            for data in components.get(target, {}).get(name_orig, []):
                if data.available:
                    data_copy = copy.deepcopy(data)
                    data_copy.name_alias = name_alias
                    data_copy.params_update = settings
                    res.append(data_copy)
                    logger.debug(f"Found {target.title()} '{name_orig}' [{data_copy.__repr__()}]")
                    break
                else:
                    logger.error(f"{target.title()} '{name_orig}' is not available")
        return res

    def check_config_issues(self, name: Literal['launcher', 'reporter']) -> None:
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
        if not launcher.get('results'):
            issues.append("No results directory specified in launcher.")
        if launcher.get('repeat') <= 0:
            issues.append("Invalid repeat number specified in launcher. Must be greater than 0.")
        if not isinstance(launcher.get('save'), bool):
            issues.append("Invalid save option specified in launcher. Must be True or False.")
        if not launcher.get('algorithms'):
            issues.append("No algorithms specified in launcher.")
        if not launcher.get('problems'):
            issues.append("No problems specified in launcher.")
        return issues

    def check_reporter_config(self) -> list[str]:
        issues = []
        reporter = self.config['reporter']
        results = reporter.get('results')
        comparisons = reporter.get('comparisons', [])
        if not results:
            issues.append("No results directory specified in reporter or launcher.")
        if not comparisons:
            issues.append("No algorithms specified in 'reporter.comparisons' or 'launcher.algorithms'")
        else:
            validate_values: list[list[str]] = []
            for item in comparisons:
                if isinstance(item, str) and item:
                    validate_values.append([item])
                elif isinstance(item, list) and item:
                    validate_values.append(item)
                else:
                    issues.append(f"Invalid value [type:{type(item)}, value:{item}] specified in reporter.")
            reporter['comparisons'] = validate_values
        return issues
