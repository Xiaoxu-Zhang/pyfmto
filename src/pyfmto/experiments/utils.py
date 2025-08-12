import copy
import numpy as np
import os
import pandas as pd
import platform
import seaborn
import shutil
import subprocess
import sys
import time
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
from pyfmto.utilities import logger, save_msgpack, titled_tabulate, load_msgpack
from pyfmto.utilities.schemas import LauncherConfig, ReporterConfig


class RunSolutions:
    def __init__(self, run_solutions: Optional[dict] = None):
        self._solutions: dict[int, dict] = {}
        if run_solutions:
            self.__dict__.update(copy.deepcopy(run_solutions))

    def update(self, cid: int, solution: Solution):
        self._solutions[cid] = copy.deepcopy(solution.to_dict())

    def get_solutions(self, cid: int) -> Solution:
        """
        Retrieve solutions for the specified client IDs.

        Parameters
        ----------
        cid : Union[int, list[int], tuple[int]]
            A single client ID (int) or a list/tuple of client IDs.

        Returns
        -------
        Solution
            - A `Solution` object if a single client ID is provided.
            - A dictionary of `Solution` objects if a list or tuple of client IDs is provided.
        """
        try:
            return Solution(self._solutions[cid])
        except KeyError:
            raise KeyError(f"Client {cid} not found in results")

    @property
    def solutions(self) -> list[Solution]:
        return [Solution(self._solutions[cid]) for cid in self.sorted_ids]

    def to_msgpack(self, filename: Union[str, Path] = 'results.msgpack'):
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


class Statistics:
    def __init__(self,
                 mean_orig: np.ndarray, mean_log: np.ndarray,
                 std_orig: np.ndarray, std_log: np.ndarray,
                 se_orig: np.ndarray, se_log: np.ndarray,
                 opt_orig: np.ndarray, opt_log: np.ndarray):
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
        self.x = np.array([])
        self.x_global = np.array([])
        self.y_global = np.array([])


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
        logger.debug("Server started.")
        time.sleep(1)
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
        thread_pool = ThreadPoolExecutor(max_workers=len(clients))
        client_futures = [thread_pool.submit(c.start) for c in clients]
        thread_pool.shutdown(wait=True)
        return [c.result() for c in client_futures]

    @staticmethod
    def kill_server():
        if platform.system() == 'Windows':
            os.system("taskkill /f /im AlgServer.exe")
        else:
            os.system("pkill -f AlgServer")
        time.sleep(.5)  # Waiting for the server termination (at least .2s)

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
    def get_runs_data(cid: int, runs_data: list[RunSolutions]):
        runs_x_data = []
        runs_y_data = []
        tmp = runs_data[0].get_solutions(cid)
        fe_init = tmp.fe_init
        fe_max = tmp.size
        x_global = tmp.x_global
        y_global = tmp.y_global
        for run_res in runs_data:
            solution = run_res.get_solutions(cid)
            runs_x_data.append(solution.x)
            runs_y_data.append(solution.y_homo_decrease)
        res = {
            'runs_x': runs_x_data,
            'runs_y': runs_y_data,
            'fe_init': fe_init,
            'fe_max': fe_max,
            'x_global': x_global,
            'y_global': y_global
        }
        return res

    @staticmethod
    def calculate_statistics(data: Union[list, np.ndarray]) -> Statistics:
        """
        Calculate statistical measures for the given data, including mean, standard deviation,
        and standard error for both the original and logarithmic scales.

        Parameters
        ----------
        data : Union[list, ndarray]
            A list or 2D NumPy array where each row represents a run and each column represents
            a fitness value of a problem.

        Returns
        -------
        Statistics
            An instance of the `Statistics` class, refer to `Statistics` for more information.

        References
        ----------
        - Paper: Streiner, D. L. (1996). Maintaining Standards: Differences between the Standard
          Deviation and Standard Error, and When to Use Each. The Canadian Journal of
          Psychiatry, 498–502. https://doi.org/10.1177/070674379604100805

        - Doc of ``np.std``: In statistics, ... of the population. The use of :math:`N-1` in the
          denominator is often called "Bessel's correction" because ..., but less than it would
          have been without the correction. For this quantity, use ``ddof=1``.
        """
        data_orig = np.array(data) if isinstance(data, list) else data
        num_runs = data_orig.shape[0]
        data_orig[data_orig < 1e-20] = 1e-20
        data_log = np.log10(data_orig)

        mean_orig = np.mean(data_orig, axis=0)
        mean_log = np.mean(data_log, axis=0)
        std_orig = np.std(data_orig, ddof=1, axis=0)
        std_log = np.std(data_log, ddof=1, axis=0)
        se_orig = std_orig / np.sqrt(num_runs)
        se_log = std_log / np.sqrt(num_runs)
        opt_orig = data_orig[:, -1]
        opt_log = data_log[:, -1]

        sta = Statistics(
            mean_orig=mean_orig,
            mean_log=mean_log,
            std_orig=std_orig,
            std_log=std_log,
            se_orig=se_orig,
            se_log=se_log,
            opt_orig=opt_orig,
            opt_log=opt_log
        )

        return sta

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
    def plotting(ax, statistics: dict[str, dict[str, Statistics]], alg, key, showing_size, in_log_scale, alpha, color):
        c_data = statistics[alg][key]
        fe_init = c_data.fe_init
        fe_max = c_data.fe_max
        size_optimization = fe_max - fe_init
        size_plot = fe_max if showing_size is None else showing_size

        x_indices = np.arange(size_plot) + 1
        avg = c_data.mean_log[-size_plot:] if in_log_scale else c_data.mean_orig[-size_plot:]
        se = c_data.se_log[-size_plot:] if in_log_scale else c_data.se_orig[-size_plot:]
        se_upper = list(map(lambda x: x[0] - x[1], zip(avg, se)))
        se_lower = list(map(lambda x: x[0] + x[1], zip(avg, se)))
        ax.plot(x_indices, avg, label=alg, color=color)
        ax.fill_between(x_indices, se_upper, se_lower, alpha=alpha, color=color)
        if size_plot > size_optimization:
            optimization_start_at = size_plot - size_optimization
        else:
            optimization_start_at = 0
        if optimization_start_at > 0:
            ax.axvline(x=optimization_start_at, color='gray', linestyle='--', label='Start index')

    @staticmethod
    def get_np_name(np_per_dim: int):
        return 'IID' if np_per_dim == 1 else f"NIID{np_per_dim}"

    @staticmethod
    def load_runs_data(path_list: Union[list[Path], list[str]]) -> list[RunSolutions]:
        return [RunSolutions(load_msgpack(p)) for p in path_list]

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
    def plot_violin(statis: Statistics, figsize, filename: Path, title: str):
        samples = statis.x
        n_dims = samples.shape[1]
        df = pd.DataFrame(samples, columns=[f'x{i + 1}' for i in range(n_dims)])
        df_melted = df.melt(var_name='Dimension', value_name='Value')
        plt.figure(figsize=figsize)
        ax = seaborn.violinplot(
            data=df_melted,
            x='Dimension',
            y='Value',
            hue='Dimension',
            inner='quartile'
        )

        x_global = statis.x_global
        if x_global is not None:
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
        file_names = [str(name) for name in file_dir.iterdir()]
        n_row, n_col = ReporterUtils.find_grid_shape(len(file_names))
        img_grid = [['' for _ in range(n_col)] for _ in range(n_row)]
        for i, p in enumerate(sorted(file_names)):
            img_grid[i // n_col][i % n_col] = str(p)
        merge_from = file_dir.name
        size_info = []
        images = []
        for row in img_grid:
            row_img: list[Optional[Image.Image]] = []
            for filename in row:
                if not filename:
                    row_img.append(None)
                    continue
                image = Image.open(filename)
                size_info.append(image.size)
                row_img.append(image)
            images.append(row_img)

        size_info = np.array(size_info, dtype=int)
        width, height = np.min(size_info, axis=0)

        canvas_width = width * n_col
        canvas_height = height * n_row

        merged = Image.new('RGB', (canvas_width, canvas_height), color='white')
        for row_id in range(n_row):
            for col_id in range(n_col):
                img = images[row_id][col_id]
                if not img:
                    continue
                merged.paste(
                    img.resize((width, height), Image.Resampling.LANCZOS),
                    (col_id * width, row_id * height)
                )
        merged.save(file_dir.parent / f'{merge_from}.png')
        if clear:
            shutil.rmtree(file_dir)
