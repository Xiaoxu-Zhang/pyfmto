import copy
import os

from numpy import ndarray
from pathlib import Path
from pyfmto.problems import Solution
from pyfmto.utilities import logger
from typing import Optional, Union

from ..algorithms import load_algorithm
from ..utilities.io import load_yaml, save_msgpack, load_msgpack

__all__ = [
    'gen_path',
    'check_path',
    'load_results',
    'save_results',
    'clear_console',
    'prepare_server',
    'load_runs_settings',
    'load_analyses_settings',
    'Statistics',
    'RunSolutions',
    'MergedSolution'
]


def clear_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def load_runs_settings():
    settings = load_yaml('settings.yaml').get('runs')
    others = settings.get('others')
    problems = settings.get('problems')
    algorithms = settings.get('algorithms')
    settings['algorithms'] = [algorithms] if isinstance(algorithms, str) else algorithms
    algorithms = settings['algorithms']

    type_errors = []
    if not isinstance(others, (dict, type(None))):
        type_errors.append(f"'others' must be a dict, got {type(others)} instead.")
    if not _is_1d_str_list(algorithms):
        type_errors.append(f"'algorithms' must be a list of strings, got {type(algorithms)} instead.")
    if not isinstance(problems, dict):
        type_errors.append(f"'problems' must be a dict, got {type(problems)} instead.")

    if any(type_errors):
        err_str = '\n'.join(type_errors)
        raise TypeError(f"Invalid settings: \n{err_str}")
    else:
        return settings


def load_analyses_settings():
    settings = load_yaml('settings.yaml').get("analyses")
    res_path = settings.get('results')
    settings['results'] = Path('out', 'results') if res_path is None else Path(res_path)

    type_errors = []
    if not _is_2d_str_list(settings.get('algorithms')):
        type_errors.append("'algorithms' must be a 2D str list")
    if not _is_1d_str_list(settings.get('problems')):
        type_errors.append("'problems' must be a list of strings")
    if not _is_1d_int_list(settings.get('np_per_dim')):
        type_errors.append("'np_per_dim' must be a list of integers")

    if any(type_errors):
        err_str = '\n'.join(type_errors)
        raise TypeError(f"Invalid settings: \n{err_str}")
    else:
        return settings


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


def prepare_server(alg_name):
    alg_conf = load_algorithm(alg_name)
    pkg_name = f"algorithms.{alg_name}"
    is_builtin = alg_conf['is_builtin_alg']
    alg_pkg = f'pyfmto.{pkg_name}' if is_builtin else pkg_name
    server_name = alg_conf['server'].__name__
    server_content = [
        f"import sys\n",
        f"from pathlib import Path\n"
        f"sys.path.append(str(Path(__file__).parent))\n",
        f"from {alg_pkg} import {server_name}\n\n\n",
        f"if __name__ == '__main__':\n",
        f"    server = {server_name}()\n",
        f"    server.start()"
    ]
    with open('temp_server.py', 'w') as f:
        f.writelines(server_content)


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
    if run_solutions.num_clients == 0:
        logger.info('No results found.')
    else:
        data = run_solutions.to_dict()
        save_msgpack(data, file_name)
        logger.info(f"Results saved to {file_name}")


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

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def clear(self):
        self._solutions = {}

    @property
    def num_clients(self):
        return len(self._solutions)

    @property
    def sorted_ids(self):
        return sorted(map(int, self._solutions.keys()))

    def _get_solution(self, cid):
        str_id = cid
        if str_id not in self._solutions:
            raise KeyError(f"Client {cid} not found in results")
        return Solution(self._solutions[str_id])


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


class MergedSolution:
    def __init__(self):
        self.merged_result = {}

    def append(self, cid: int, sta: Statistics):
        self.merged_result[cid] = sta

    def get_data(self, cid: int) -> Statistics:
        return self.merged_result[cid]

    @property
    def sorted_ids(self) -> list[int]:
        return sorted(self.merged_result.keys())

    @property
    def num_clients(self) -> int:
        return len(self.merged_result)
