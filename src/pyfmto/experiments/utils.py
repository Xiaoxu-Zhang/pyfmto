import copy
import os
from itertools import product
from numpy import ndarray
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import Optional, Union

from pyfmto.problems import Solution
from pyfmto.utilities import logger, save_msgpack, load_msgpack


class LauncherConfig(BaseModel):
    results: str = 'out/results'
    repeat: int = 1
    seed: int = 42
    backup: bool = True
    save: bool = True
    algorithms: list[str]
    problems: list[str]

    @field_validator('results')
    def results_must_be_not_none(cls, v):
        return v if v is not None else 'out/results'

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


class ReporterConfig(BaseModel):
    results: str = 'out/results'
    algorithms: list[list[str]]
    problems: list[str]

    @field_validator('results')
    def results_must_be_not_none(cls, v):
        return v if v is not None else 'out/results'

    @field_validator('algorithms')
    def inner_lists_must_have_min_length(cls, v):
        for inner_list in v:
            if len(inner_list) < 2:
                raise ValueError('inner lists must have at least 2 elements')
        return v

    @field_validator('problems', 'algorithms')
    def outer_list_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('problems list must have at least 1 element')
        return v


def clear_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def kill_server():
    if os.name == 'win32':
        os.system("taskkill /f /im AlgServer.exe")
    else:
        os.system("pkill -f AlgServer")


def gen_exp_combinations(launcher_conf: LauncherConfig, alg_conf: dict, prob_conf: dict):
    alg_items = [(name, alg_conf.get(name, {})) for name in launcher_conf.algorithms]
    prob_conf = {name: prob_conf.get(name, {}) for name in launcher_conf.problems}
    prob_items = _combine_args(prob_conf)
    combinations = [(*alg_item, *prob_item) for alg_item, prob_item in product(alg_items, prob_items)]
    return combinations


def _combine_args(args: dict):
    prob_items = []
    for prob_name, prob_args in args.items():
        list_args = {k: v for k, v in prob_args.items() if isinstance(v, list)}
        non_list_args = {k: v for k, v in prob_args.items() if not isinstance(v, list)}

        if not list_args:
            args = {**non_list_args}
            prob_items.append((prob_name, args))
            continue

        keys, values = zip(*list_args.items())
        for combination in product(*values):
            args = {**non_list_args, **dict(zip(keys, combination))}
            prob_items.append((prob_name, args))
    return prob_items


def parse_reporter_config(config: dict, prob_conf: dict):
    reporter = ReporterConfig(**config)
    algorithms = reporter.algorithms
    problems = reporter.problems
    prob_items = []
    for name, args in _combine_args({name: prob_conf.get(name, {}) for name in problems}):
        src_prob = args.get('src_problem')
        np_per_dim = args.get('np_per_dim', 1)
        if src_prob:
            prob_items.append((f"{name.upper()}-{src_prob}", np_per_dim))
        else:
            prob_items.append((name.upper(), np_per_dim))
    analysis_comb = [(alg, *prob_item) for alg, prob_item in product(algorithms, prob_items)]
    initialize_comb = [(alg, *prob_item) for alg, prob_item in product(sum(algorithms, []), prob_items)]
    analyses = {
        'results': reporter.results,
        'analysis_comb': analysis_comb,
        'initialize_comb': initialize_comb
    }
    return analyses


def _is_2d_str_list(lst):
    if not isinstance(lst, list):
        return False
    elif not all(isinstance(sub_lst, list) for sub_lst in lst):
        return False
    else:
        for sub_lst in lst:
            for item in sub_lst:
                if not isinstance(item, str):
                    return False
    return True


def _is_1d_str_list(lst):
    return isinstance(lst, list) and all(isinstance(item, str) for item in lst)


def _is_1d_int_list(lst):
    return isinstance(lst, list) and all(isinstance(item, int) for item in lst)


def gen_path(alg_name, prob_name, prob_args):
    np_per_dim = prob_args.get('np_per_dim')
    dim = prob_args.get('dim')
    dim = '' if dim is None else f'_{dim}D'
    src_prob = prob_args.get('src_problem', '')
    src_prob = f'-{src_prob}' if src_prob != '' else ''
    init_data_type = 'IID' if np_per_dim in (1, None) else f'NIID{np_per_dim}'
    res_root = Path('out', 'results', alg_name, f"{prob_name.upper()}{src_prob}{dim}", init_data_type)
    return res_root


def check_path(res_root):
    res_root = Path(res_root)
    if not res_root.exists():
        res_root.mkdir(parents=True)
        return 0
    else:
        return len(os.listdir(res_root))


def save_results(clients_res, res_path, curr_run):
    res_path = Path(res_path)
    file_name = res_path / f"Run {curr_run}.msgpack"
    run_solutions = RunSolutions()
    for cid, solution in clients_res:
        run_solutions.update(cid, solution)
    run_solutions.to_msgpack(file_name)


def load_results(file_name):
    data = load_msgpack(file_name)
    return RunSolutions(data)


class RunSolutions:
    def __init__(self, run_solutions: Optional[dict] = None):
        self._solutions: dict[int, dict] = {}
        if run_solutions:
            self.__dict__.update(copy.deepcopy(run_solutions))

    def update(self, cid: int, solution: Solution):
        self._solutions[cid] = copy.deepcopy(solution.to_dict())

    def get_solutions(self, ids: Union[int, list[int], tuple[int]]) -> Union[Solution, dict[int, Solution]]:
        """
        Retrieve solutions for the specified client IDs.

        Parameters
        ----------
        ids : Union[int, list[int], tuple[int]]
            A single client ID (int) or a list/tuple of client IDs.

        Returns
        -------
        Union[Solution, dict[int, Solution]]
            - A `Solution` object if a single client ID is provided.
            - A dictionary of `Solution` objects if a list or tuple of client IDs is provided.
        """
        if not isinstance(ids, (list, tuple)):
            return self._get_solution(int(ids))
        else:
            solutions = {}
            for cid in ids:
                solutions[cid] = self._get_solution(cid)
            return solutions

    @property
    def solutions(self) -> list[Solution]:
        return [Solution(self._solutions[cid]) for cid in self.sorted_ids]

    def to_msgpack(self, filename: Union[str, Path]='results.msgpack'):
        if self.num_clients == 0:
            logger.info('Empty RunSolutions.')
        else:
            data = copy.deepcopy(self.__dict__)
            save_msgpack(data, filename)
            logger.info(f"Results saved to {filename}")

    def clear(self):
        self._solutions = {}

    @property
    def num_clients(self):
        return len(self._solutions)

    @property
    def sorted_ids(self):
        return sorted(map(int, self._solutions.keys()))

    def _get_solution(self, cid: int):
        try:
            return Solution(self._solutions[cid])
        except KeyError:
            raise KeyError(f"Client {cid} not found in results")


class Statistics:
    def __init__(self,
                 mean_orig: ndarray, mean_log: ndarray,
                 std_orig: ndarray, std_log: ndarray,
                 se_orig: ndarray, se_log: ndarray,
                 opt_orig: ndarray, opt_log: ndarray):
        """
        Initialize the Statistics class instance with statistical measures for both original and logarithmic scales.

        Parameters
        ----------
        mean_orig : ndarray
            Mean values of multiple runs in the original scale.
        mean_log : ndarray
            Mean values of multiple runs in the logarithmic scale.
        std_orig : ndarray
            Standard deviations of multiple runs in the original scale.
        std_log : ndarray
            Standard deviations of multiple runs in the logarithmic scale.
        se_orig : ndarray
            Standard errors of multiple runs in the original scale.
        se_log : ndarray
            Standard errors of multiple runs in the logarithmic scale.
        opt_orig : ndarray
            Optimal values of multiple runs in the original scale.
        opt_log : ndarray
            Optimal values of multiple runs in the logarithmic scale.
        """
        self.mean_orig = mean_orig
        self.mean_log = mean_log
        self.std_orig = std_orig
        self.std_log = std_log
        self.se_orig = se_orig
        self.se_log = se_log
        self.opt_orig = opt_orig
        self.opt_log = opt_log
        self.fe_init = 0
        self.fe_max = 0
        self.x = None
        self.x_global = None
        self.y_global = None
