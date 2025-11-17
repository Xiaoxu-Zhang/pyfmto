import copy
import importlib
import inspect
import numpy as np
import os
import pandas as pd
import seaborn
import shutil
import subprocess
import sys
import time
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from PIL import Image
from matplotlib import pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Optional, Union, Type

from pyfmto.framework import Client, Server
from pyfmto.problems import Solution
from pyfmto.utilities import logger, save_msgpack, titled_tabulate, load_msgpack, parse_yaml, colored
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig

StatisData = namedtuple("StatisData", ['mean', 'std', 'se', 'opt'])

__all__ = [
    "MetaData",
    "RunSolutions",
    "ClientDataStatis",
    "MergedResults",
    "LauncherUtils",
    "ReporterUtils",
    "Algorithm",
    "load_algorithm",
    "list_algorithms",
    "get_alg_kwargs"
]


class RunSolutions:
    def __init__(self, run_solutions: Optional[dict] = None):
        self._solutions: dict[int, dict] = {}
        if run_solutions:
            self.__dict__.update(copy.deepcopy(run_solutions))

    def update(self, cid: int, solution: Solution):
        self._solutions[cid] = copy.deepcopy(solution.to_dict())

    @property
    def solutions(self) -> list[Solution]:
        return [Solution(self._solutions[cid]) for cid in self.sorted_ids]

    def __getitem__(self, item):
        try:
            return Solution(self._solutions[item])
        except KeyError:
            raise KeyError(f"Client id '{item}' not found in results")

    def items(self) -> list[tuple[int, Solution]]:
        return list(zip(self.sorted_ids, self.solutions))

    def to_msgpack(self, filename: Union[str, Path] = 'results.msgpack'):
        if self.num_clients == 0:
            raise ValueError('Empty RunSolutions')
        else:
            data = copy.deepcopy(self.__dict__)
            save_msgpack(data, filename)
            logger.info(f"Results saved to {filename}")

    @property
    def num_clients(self):
        return len(self._solutions)

    @property
    def sorted_ids(self):
        return sorted(map(int, self._solutions.keys()))


class ClientDataStatis:
    fe_init: int
    fe_max: int
    lb: np.ndarray
    ub: np.ndarray
    x_init: np.ndarray
    y_init: np.ndarray
    x_alg: np.ndarray
    y_alg: np.ndarray
    y_dec_mat: np.ndarray
    y_inc_mat: np.ndarray
    x_global: np.ndarray
    y_global: np.ndarray
    y_mean: np.ndarray

    def __init__(self, solutions: list[Solution]):
        if len(solutions) < 1:
            raise ValueError('Empty list of Solutions')
        self._preprocessing(solutions)

    def _preprocessing(self, solutions):
        x = []
        y_dec = []
        y_inc = []
        x_init = []
        y_init = []
        x_alg = []
        y_alg = []
        for solution in solutions:
            x.append(solution.x)
            y_dec.append(solution.y_homo_decrease)
            y_inc.append(solution.y_homo_increase)
            x_init.append(solution.x_init)
            y_init.append(solution.y_init)
            x_alg.append(solution.x_alg)
            y_alg.append(solution.y_alg)
        self.x_init = np.vstack(x_init)
        self.y_init = np.vstack(y_init)
        self.x_alg = np.vstack(x_alg)
        self.y_alg = np.vstack(y_alg)
        self.y_dec_mat = np.array(y_dec)
        self.y_inc_mat = np.array(y_inc)
        self.y_dec_mat[self.y_dec_mat < 1e-20] = 1e-20
        self.y_inc_mat[self.y_inc_mat < 1e-20] = 1e-20
        self.x_global = solutions[0].x_global
        self.y_global = solutions[0].y_global
        self.fe_init = solutions[0].fe_init
        self.fe_max = solutions[0].fe_max
        self.lb = solutions[0].lb
        self.ub = solutions[0].ub

    @property
    def is_known_optimal(self):
        return self.x_global.size > 0

    @property
    def y_dec_statis(self) -> StatisData:
        return ReporterUtils.get_mat_statis(self.y_dec_mat)

    @property
    def y_inc_statis(self) -> StatisData:
        return ReporterUtils.get_mat_statis(self.y_inc_mat)

    @property
    def y_dec_log_statis(self) -> StatisData:
        return ReporterUtils.get_mat_statis(np.log10(self.y_dec_mat))

    @property
    def y_inc_log_statis(self) -> StatisData:
        return ReporterUtils.get_mat_statis(np.log10(self.y_inc_mat))


class MergedResults:

    def __init__(self, runs_data: list[RunSolutions]):
        if len(runs_data) < 1:
            raise ValueError('Empty list of RunSolutions')
        self._original: list[RunSolutions] = runs_data
        self._reorganized: dict[int, list[Solution]] = defaultdict(list)
        self._merged: dict[str, ClientDataStatis] = {}
        self._reorganizing()
        self._merging()

    def _reorganizing(self):
        for run_data in self._original:
            for cid, solution in run_data.items():
                self._reorganized[cid].append(solution)

    def _merging(self):
        self._merged = {
            f"Client {cid:>02d}": ClientDataStatis(solutions)
            for cid, solutions in self._reorganized.items()
        }

    def get_statis(self, c_name) -> ClientDataStatis:
        return self._merged[c_name]

    def items(self) -> list[tuple[str, ClientDataStatis]]:
        return list(self._merged.items())

    @property
    def num_runs(self):
        return len(self._original)

    @property
    def dim(self):
        return self._original[0].solutions[0].dim

    @property
    def sorted_names(self):
        return sorted(self._merged.keys())


class MetaData:

    def __init__(
            self,
            data: dict[str, MergedResults],
            problem: str,
            npd_name: str,
            filedir: Path):
        self._data = data
        self.problem = problem
        self.npd_name = npd_name
        self.filedir = filedir

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def items(self) -> list[tuple[str, MergedResults]]:
        return list(self._data.items())

    @property
    def clt_names(self):
        if self.alg_num == 0:
            return []
        else:
            return list(self._data.values())[0].sorted_names

    @property
    def clt_num(self):
        return len(self.clt_names)

    @property
    def alg_num(self):
        return len(self._data)

    @property
    def dim(self):
        if self.alg_num == 0:
            return 0
        else:
            return list(self._data.values())[0].dim

    @property
    def num_runs(self):
        return list(self._data.values())[0].num_runs

    @property
    def alg_names(self):
        return list(self._data.keys())


class LauncherUtils:

    @staticmethod
    def terminate_popen(process: subprocess.Popen):
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    @contextmanager
    def running_server(server: Type[Server], **kwargs):
        """
        Start the server in a subprocess with a context manager approach.
        The server will be automatically terminated when exiting the context.

        Parameters
        ----------
        server:
            The server class itself, not an instance.
        kwargs:
            The kwargs of the server.

        Example
        -------
        with LauncherUtils.start_server_context(MyServer, port=8080) as server_process:
            # Do something with the server
            pass
        # Server is automatically terminated here
        """
        module_name = server.__module__
        class_name = server.__name__

        cmd = [
            sys.executable, "-c",
            f"from {module_name} import {class_name}; "
            f"srv = {class_name}(**{repr(kwargs)}); "
            f"srv.start()"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Server started.")
        time.sleep(3)
        try:
            yield process
        finally:
            LauncherUtils.terminate_popen(process)
            logger.debug("Server terminated.")

    @staticmethod
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
        pool = ThreadPoolExecutor(max_workers=len(clients))
        futures = [pool.submit(c.start) for c in clients]
        pool.shutdown(wait=True)
        return [fut.result() for fut in futures]

    @staticmethod
    def gen_exp_combinations(launcher_conf: LauncherConfig, alg_conf: dict, prob_conf: dict):
        alg_items = [(name, alg_conf.get(name, {})) for name in launcher_conf.algorithms]
        prob_conf = {name: prob_conf.get(name, {}) for name in launcher_conf.problems}
        prob_items = LauncherUtils.combine_args(prob_conf)
        combinations = [(*alg_item, *prob_item) for alg_item, prob_item in product(alg_items, prob_items)]
        return combinations

    @staticmethod
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

    @staticmethod
    def gen_path(alg_name, prob_name, prob_args):
        np_per_dim = prob_args.get('np_per_dim')
        dim = prob_args.get('dim')
        dim = '' if dim is None else f'_{dim}D'
        src_prob = prob_args.get('src_problem', '')
        src_prob = f'-{src_prob}' if src_prob != '' else ''
        init_data_type = 'IID' if np_per_dim in (1, None) else f'NIID{np_per_dim}'
        res_root = Path('out', 'results', alg_name, f"{prob_name.upper()}{src_prob}{dim}", init_data_type)
        return res_root


class ReporterUtils:

    @staticmethod
    def check_rows(data_list: list[list], col_title: list[str], row_title: list[str], msg=''):
        try:
            data_mat = np.array(data_list, dtype=int)
        except Exception:
            tab = titled_tabulate(msg, '=', data_list, headers=col_title, showindex=row_title, tablefmt='psql')
            raise ValueError(f"\n{tab}")
        if not np.all(np.equal(data_mat, data_mat[0])):
            tab = titled_tabulate(msg, '=', data_list, headers=col_title, showindex=row_title, tablefmt='psql')
            raise ValueError(f"\n{tab}")

    @staticmethod
    def find_grid_shape(size):
        """
        Determine the shape of a grid (a, b) for a given size such that:
        1. a * b >= size
        2. a * b - size is minimized
        3. a / b is as close to 1 as possible

        Parameters
        ----------
        size : int
            The total number of elements in the grid.

        Returns
        -------
        tuple
            A tuple (a, b) representing the dimensions of the grid.
            a and b are integers such that a * b >= size and a / b is minimized.
        """
        if size <= 0:
            raise ValueError("Size must be a positive integer.")
        a = int(np.sqrt(size))
        b = size // a
        while a * b < size:
            if a < b:
                a += 1
            else:
                b += 1

        return (a, b) if a > b else (b, a)

    @staticmethod
    def parse_reporter_config(config: dict, prob_conf: dict):
        reporter = ReporterConfig(**config)
        algorithms = reporter.algorithms
        problems = reporter.problems
        prob_items = []
        for name, args in LauncherUtils.combine_args({name: prob_conf.get(name, {}) for name in problems}):
            src_prob = args.get('src_problem')
            npd_name = ReporterUtils.get_np_name(args.get('np_per_dim', 1))
            if src_prob:
                prob_items.append((f"{name.upper()}-{src_prob}", npd_name))
            else:
                prob_items.append((name.upper(), npd_name))
        analysis_comb = [(alg, *prob_item) for alg, prob_item in product(algorithms, prob_items)]
        initialize_comb = [(alg, *prob_item) for alg, prob_item in product(sum(algorithms, []), prob_items)]
        analyses = {
            'results': reporter.results,
            'analysis_comb': analysis_comb,
            'initialize_comb': initialize_comb
        }
        return analyses

    @staticmethod
    def get_t_test_suffix(opt_list1, opt_list2, mean1, mean2, pvalue):
        diff = mean1 - mean2
        _, p = stats.ttest_ind(opt_list1, opt_list2)
        if p > pvalue:
            suffix = "≈"
        elif diff > 0:
            suffix = "-"
        else:
            suffix = "+"
        return suffix

    @staticmethod
    def plotting(ax, merged_results: MergedResults, alg_name, c_name, showing_size, in_log_scale, alpha, color):
        c_data = merged_results.get_statis(c_name)
        fe_init = c_data.fe_init
        fe_max = c_data.fe_max
        size_optimization = fe_max - fe_init
        size_plot = fe_max if showing_size == -1 else showing_size
        x_indices: np.ndarray = np.arange(size_plot) + 1
        avg = c_data.y_dec_log_statis.mean[-size_plot:] if in_log_scale else c_data.y_dec_statis.mean[-size_plot:]
        se = c_data.y_dec_log_statis.se[-size_plot:] if in_log_scale else c_data.y_dec_statis.se[-size_plot:]
        se_upper = list(map(lambda x: x[0] - x[1], zip(avg, se)))
        se_lower = list(map(lambda x: x[0] + x[1], zip(avg, se)))
        ax.plot(x_indices, avg, label=alg_name, color=color)
        ax.fill_between(x_indices, se_upper, se_lower, alpha=alpha, color=color)
        if size_plot > size_optimization:
            return size_plot - size_optimization
        else:
            return 0

    @staticmethod
    def get_np_name(np_per_dim: int):
        return 'IID' if np_per_dim == 1 else f"NIID{np_per_dim}"

    @staticmethod
    def load_runs_data(file_dir: Path) -> list[RunSolutions]:
        if file_dir.exists():
            filenames = [
                file_dir / f_name
                for f_name in os.listdir(file_dir) if f_name.endswith('.msgpack')
            ]
            return [RunSolutions(load_msgpack(p)) for p in filenames]
        else:
            print(f"Result file not found in {colored(str(file_dir), 'red')}")
            return []

    @staticmethod
    def get_optimality_index_mat(mean_dict):
        df = pd.DataFrame.from_dict(mean_dict)
        data_mat = df.to_numpy()

        mat_shape = data_mat.shape[0], data_mat.shape[1]
        solo_index_mat = np.zeros(shape=mat_shape, dtype=bool)
        global_index_mat = np.zeros(shape=mat_shape, dtype=bool)

        row, col = solo_index_mat.shape
        min_idx = np.argmin(data_mat, axis=1)
        for i in range(row):
            global_index_mat[i, min_idx[i]] = True
            obj_val = data_mat[i, -1]
            for j in range(col - 1):
                if data_mat[i, j] < obj_val:
                    solo_index_mat[i, j] = True
        solo_index_mat = global_index_mat + solo_index_mat

        # row+1: the last row is the count of best solution of each algorithm
        # col+1: the first col is the index of Client
        add_row = np.zeros(shape=solo_index_mat.shape[1], dtype=bool)
        solo_index_mat = np.vstack((solo_index_mat, add_row))
        global_index_mat = np.vstack((global_index_mat, add_row))

        add_col = np.zeros(shape=solo_index_mat.shape[0], dtype=bool).reshape(-1, 1)
        solo_index_mat = np.hstack((add_col, solo_index_mat))
        global_index_mat = np.hstack((add_col, global_index_mat))
        return global_index_mat, solo_index_mat

    @staticmethod
    def get_mat_statis(y_mat: np.ndarray):
        rows = y_mat.shape[0]
        mean = np.mean(y_mat, axis=0)
        std = np.std(y_mat, ddof=1, axis=0)
        se = std / np.sqrt(rows)
        opt = y_mat[:, -1]
        return StatisData(mean, std, se, opt)

    @staticmethod
    def plot_violin(statis: ClientDataStatis, figsize, filename: Path, title: str, dpi: float):
        n_dims = statis.x_init.shape[1]
        df_init = pd.DataFrame(statis.x_init, columns=[f'x{i + 1}' for i in range(n_dims)])
        df_optimized = pd.DataFrame(statis.x_alg, columns=[f'x{i + 1}' for i in range(n_dims)])
        hue = 'X From'
        df_init[hue] = 'Init'
        df_optimized[hue] = 'Alg'
        df_combined = pd.concat([df_init, df_optimized], ignore_index=True)
        df_melted = df_combined.melt(id_vars=[hue], var_name='Dimension', value_name='Value')
        plt.figure(figsize=figsize, dpi=dpi)
        ax = seaborn.violinplot(
            data=df_melted,
            x='Dimension',
            y='Value',
            hue=hue,
            palette=['#ff7f50', '#2d98da'],
            split=True,
            inner='quartile',
            linewidth=0.5
        )
        x_global = statis.x_global
        if x_global.size > 0:
            for dim in range(n_dims):
                ax.plot(dim, x_global[dim], 'r*', markersize=8, markeredgecolor='w', linewidth=0.5)
        plt.title(title)
        plt.xlabel('Dimension Index')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def merge_images_in(file_dir: Path, clear: bool):
        file_paths = list(file_dir.iterdir())
        suffix = file_paths[0].suffix
        n_row, n_col = ReporterUtils.find_grid_shape(len(file_paths))
        merge_from = file_dir.name
        sorted_paths = sorted(str(path) for path in file_paths)
        images = [Image.open(path) for path in sorted_paths]
        sizes = np.array([img.size for img in images])
        width, height = np.min(sizes, axis=0)
        canvas_width = width * n_col
        canvas_height = height * n_row
        merged = Image.new('RGB', (canvas_width, canvas_height), color='white')

        for i, img in enumerate(images):
            row_id = i // n_col
            col_id = i % n_col
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            merged.paste(resized_img, (col_id * width, row_id * height))

        merged.save(file_dir.parent / f'{merge_from}{suffix}')
        if clear:
            shutil.rmtree(file_dir)

    @staticmethod
    def check_suffix(suffix: str, merge: bool):
        if merge and (suffix not in ['.png', '.jpg']):
            print(f"Only support suffix {colored('.png or .jpg', 'green')} "
                  f"when {colored('merge is True', 'green')}, defaulted to '.png'")
            return '.png'
        else:
            return suffix


class Algorithm:
    client: Type[Client]
    server: Type[Server]

    @property
    def is_complete(self) -> bool:
        return hasattr(self, 'client') and hasattr(self, 'server')


def load_algorithm(name: str):
    alg_dir = Path().cwd() / 'algorithms' / name
    if alg_dir.exists():
        module = importlib.import_module(f"algorithms.{name}")
    elif alg_dir.parent.exists():
        raise ValueError(f"algorithm {name} not found in {alg_dir.parent}.")
    else:
        raise FileNotFoundError(f"'algorithms' folder not found in {Path().cwd()}.")
    alg = Algorithm()

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if inspect.isclass(attr):
            if issubclass(attr, Client):
                alg.client = attr
            if issubclass(attr, Server):
                alg.server = attr
        if alg.is_complete:
            break

    msg: list[str] = [f'load algorithm {name} failed:']
    if not hasattr(alg, 'client'):
        msg.append("  Client not found.")
    if not hasattr(alg, 'server'):
        msg.append("  Server not found.")
    if len(msg) > 1:
        raise ModuleNotFoundError('\n'.join(msg))
    return alg


def list_algorithms(print_it=False):
    alg_dir = Path().cwd() / 'algorithms'
    if alg_dir.exists():
        folders = os.listdir(alg_dir)
        alg_names = [alg_name for alg_name in folders if alg_name.isupper()]
    else:
        alg_names = []
    if print_it:
        if alg_dir.exists():
            alg_str = '\n'.join(alg_names)
            print(f"Found {len(alg_names)} algorithms: \n{alg_str}")
        else:
            print(f"'algorithms' folder not found in {alg_dir.parent}.")
    return alg_names


def get_alg_kwargs(name: str):
    alg_data = load_algorithm(name)
    clt = alg_data.client.__doc__
    srv = alg_data.server.__doc__
    data = {}
    if clt:
        data.update({'client': parse_yaml(clt)})
    if srv:
        data.update({'server': parse_yaml(srv)})
    return data
