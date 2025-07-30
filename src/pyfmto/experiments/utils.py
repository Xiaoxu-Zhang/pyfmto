import copy
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from numpy import ndarray
from pathlib import Path
from typing import Optional, Union, Type

from pyfmto.framework import Client, Server
from pyfmto.problems import Solution
from pyfmto.utilities import logger, save_msgpack
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig


def clear_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def start_server(server: Type[Server], **kwargs):
    """
    Start the server in a subprocess

    Parameters
    ----------
    server:
        The server class itself, not an instance of server.
    kwargs:
        The kwargs of the server.
    """
    module_name = server.__module__
    class_name = server.__name__

    cmd = [
        "python", "-c",
        f"from {module_name} import {class_name}; "
        f"srv = {class_name}(**{repr(kwargs)}); "
        f"srv.start()"
    ]

    if os.name == 'posix':
        subprocess.Popen(cmd,
                         start_new_session=True,
                         stdin=subprocess.DEVNULL)
    elif os.name == 'nt':
        subprocess.Popen(cmd,
                         creationflags=subprocess.CREATE_NEW_CONSOLE,
                         stdin=subprocess.DEVNULL)
    else:
        raise OSError(f"Unsupported operating system: {os.name}")
    logger.info("Server started.")
    time.sleep(2)


def start_clients(clients: list[Client]) -> list[tuple[int, Solution]]:
    """
    Start the client and submit to the threadpool.

    Parameters
    ----------
    clients:
        List of Client instances.

    Returns
    -------
        List of clients results
    """
    thread_pool = ThreadPoolExecutor(max_workers=len(clients))
    client_futures = [thread_pool.submit(c.start) for c in clients]
    thread_pool.shutdown(wait=True)
    return [c.result() for c in client_futures]


def kill_server():
    if os.name == 'win32':
        os.system("taskkill /f /im AlgServer.exe")
    else:
        os.system("pkill -f AlgServer")


def gen_exp_combinations(launcher_conf: LauncherConfig, alg_conf: dict, prob_conf: dict):
    alg_items = [(name, alg_conf.get(name, {})) for name in launcher_conf.algorithms]
    prob_conf = {name: prob_conf.get(name, {}) for name in launcher_conf.problems}
    prob_items = combine_args(prob_conf)
    combinations = [(*alg_item, *prob_item) for alg_item, prob_item in product(alg_items, prob_items)]
    return combinations


def combine_args(args: dict):
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
    for name, args in combine_args({name: prob_conf.get(name, {}) for name in problems}):
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


def gen_path(alg_name, prob_name, prob_args):
    np_per_dim = prob_args.get('np_per_dim')
    dim = prob_args.get('dim')
    dim = '' if dim is None else f'_{dim}D'
    src_prob = prob_args.get('src_problem', '')
    src_prob = f'-{src_prob}' if src_prob != '' else ''
    init_data_type = 'IID' if np_per_dim in (1, None) else f'NIID{np_per_dim}'
    res_root = Path('out', 'results', alg_name, f"{prob_name.upper()}{src_prob}{dim}", init_data_type)
    return res_root


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
            raise ValueError('Empty RunSolutions')
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
