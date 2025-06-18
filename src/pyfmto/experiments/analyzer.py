import copy
import itertools
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scienceplots  # Do not remove this import
import seaborn
import time
from numpy import ndarray
from pathlib import Path
from pyfmto.utilities import logger, reset_log
from scipy import stats
from tabulate import tabulate
from tqdm import tqdm
from typing import Union, Optional

from .utils import load_results, RunSolutions, Statistics, load_analyses_settings, clear_console

T_Sta = dict[str, dict[str, Statistics]]

__all__ = ['AnalyzeResults']


class Analyzer:
    """
    Note:
        1. Please make sure your result organized in the following way
        2. String with <...> format should be replace by corresponding information

    >>> results/
    ...    │
    ...    ├── <Algor1>/
    ...    │      │
    ...    │      ├── <Prob1>/
    ...    │      │      │
    ...    │      │      ├── IID/
    ...    │      │      │    ├── <Run1>.msgpack
    ...    │      │      │    ├── <Run2>.msgpack
    ...    │      │      │    └── ...
    ...    │      │      ├── NIID2/
    ...    │      │      ├── NIID4/
    ...    │      │      └── NIID6/
    ...    │      ├── <Prob2>/
    ...    │      │      │
    ...    │      │      ├── IID/
    ...    │      │      ├── NIID2/
    ...    │      │      └── .../
    ...    │      └── <Prob...>/
    ...    ├── <Algor2>/
    ...    └── <Algor...>/
    """

    def __init__(self, root: str):
        self._root = Path(root)
        self._cache = defaultdict(dict)
        reset_log()

    def init_data(
            self,
            algorithms: list[str],
            problems: list[str],
            np_list: list[int]):
        combinations = list(itertools.product(algorithms, problems, np_list))
        for comb in combinations:
            success = self._merge_one_algorithm(*comb)
            if not success:
                logger.error(f"failed to merge {'+'.join(comb)}")

    def to_curve(
            self,
            algorithms: list[str],
            problem: str,
            np_per_dim: int,
            aspect_ratio:tuple[float, float, float]=(4,3,3),
            alpha: float=0.2,
            suffix: str='png',
            sciplot_style:Union[list[str], tuple[str]]=('science', 'ieee', 'no-latex'),
            showing_size: int=None,
            quality: Optional[int]=None,
            all_in_one: bool=True,
            in_log_scale: bool=False):
        """
        Generate and save plots of performance curves for specified algorithms and problem settings.

        Parameters
        ----------
        algorithms : list[str]
            List of algorithm names to be plotted.
        problem : str
            List of algorithm names, if len>1, squash clients data for each problem.
        np_per_dim : int
            Number of partitions per dimension.
        aspect_ratio : tuple[float, float, float], optional
            Tuple controlling the width-to-height ratio and initial size of the plot. The third value is the scaling factor.
        alpha : float, optional
            Transparency of the Standard Error region, ranging from 0 (completely transparent) to 1 (completely opaque).
        suffix : str, optional
            File format suffix for the output image. Supported formats include 'png', 'jpg', 'eps', 'svg', 'pdf'.
        sciplot_style : Union[list[str], tuple[str]], optional
            SciencePlots style parameters. Refer to the SciencePlots documentation for available styles.
        showing_size : int, optional
            Number of data points to use for plotting, taken from the last `showing_size` iterations of the convergence sequence.
            If None, use `6 * dim`.
        quality : Optional[int], optional
            Image quality parameter, affecting the quality of scalar images. Valid values are integers from 1 to 9.
            If None, the default figure DPI is used.
        all_in_one : bool, optional
            If True, all curves are plotted on a single figure. If False, each client's curves are plotted in separate figures.
        in_log_scale : bool, optional
            If True, the plot is generated on a logarithmic scale. If False, the plot is generated on an original scale.

        Returns
        -------
        None
            The function saves the generated plots to the specified file path(s) without returning any value.
        """
        # Prepare the data
        statistics = self._get_statistics(algorithms, problem, np_per_dim)
        algorithms = list(statistics.keys()) # Get the list of available algorithms
        keys = list(list(statistics.values())[0].keys())
        file_name = self._gen_name(algorithms[-1], problem, np_per_dim, keys, suffix, single_file=all_in_one, in_log_scale=in_log_scale)
        row, col = self._find_grid_shape(len(keys))
        fig_size = self._aspect_ratio(*aspect_ratio)
        dpi = 100 * quality if quality in range(10) else 'figure'

        # Plot the data
        colors = seaborn.color_palette("bright", len(algorithms) - 1).as_hex()
        colors.append("#ff0000")
        with plt.style.context(sciplot_style):
            if all_in_one:
                fig, axes = plt.subplots(row, col, figsize=fig_size)
                for idx, key in enumerate(keys):
                    ax = axes[idx // col, idx % col]
                    for alg, color in zip(algorithms, colors):
                        opt_start_at = self._plotting(ax, statistics, alg, key, showing_size, in_log_scale, alpha, color)
                        ax.set_title(key)
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Fitness')
                    if opt_start_at > 0:
                        ax.axvline(x=opt_start_at, color='gray', linestyle='--', label='Start index')
                    ax.legend()
                fig.tight_layout()
                plt.savefig(file_name, dpi=dpi)
            else:
                for key in keys:
                    for alg, color in zip(algorithms, colors):
                        opt_start_at = self._plotting(plt, statistics, alg, key, showing_size, in_log_scale, alpha, color)
                        plt.title(key)
                        plt.xlabel('Iteration')
                        plt.ylabel('Fitness')
                    if opt_start_at > 0:
                        plt.axvline(x=opt_start_at, color='gray', linestyle='--', label='Start index')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(file_name[key])
                    plt.close()

    def to_excel(
            self,
            algorithms: list[str],
            problem: str,
            np_per_dim: int,
            threshold_p: float=0.05,
            styles: Union[list[str], tuple[str]]=('color-bg-grey', 'style-font-bold', 'style-font-underline')):
        """
        Generate and save an Excel file containing performance statistics for specified algorithms and problem settings.

        Parameters
        ----------
        algorithms : list[str]
            List of algorithm names to be included in the analysis.
        problem : str
            Name of the problem being analyzed.
        np_per_dim : int
            Number of partitions per dimension.
        threshold_p : float, optional
            T-test threshold parameter to determine statistical significance. Default is 0.05.
        styles : Union[list[str], tuple[str]], optional
            List or tuple of style parameters to apply to the Excel cells.

        Notes
        -----
            The following styles are supported for personalized excel style:
             - ``color-bg-[red|grey|green|blue|yellow|purple|orange|pink]`` ---Background colors (only one can be applied, supported colors are in [])
             - ``color-font-[red|green|blue|yellow|purple|orange|pink]`` ---Font colors (only one can be applied, supported colors are in [])
             - ``type-font-[bold|italic|underline]`` ---Font types (multiple can be applied, supported types are in [])

        Returns
        -------
        None
            The function saves the generated Excel file to the specified file path without returning any value.
        """
        style_map = {
            "color-bg-red": "background-color: #ff0000",
            "color-bg-grey": "background-color: #c0c0c0",
            "color-bg-green": "background-color: #00ff00",
            "color-bg-blue": "background-color: #0000ff",
            "color-bg-yellow": "background-color: #ffff00",
            "color-bg-purple": "background-color: #800080",
            "color-bg-orange": "background-color: #ffa500",
            "color-bg-pink": "background-color: #ffc0cb",
            "color-font-red": "color: #ff0000",
            "color-font-green": "color: #00ff00",
            "color-font-blue": "color: #0000ff",
            "color-font-yellow": "color: #ffff00",
            "color-font-purple": "color: #800080",
            "color-font-orange": "color: #ffa500",
            "color-font-pink": "color: #ffc0cb",
            "style-font-bold": "font-weight: bold",
            "style-font-italic": "font-style: italic",
            "style-font-underline": "text-decoration: underline"
        }
        df_data = self._get_table_df(algorithms, problem, np_per_dim, threshold_p)
        global_df, solo_df, global_index_mat, solo_index_mat, file_name = df_data
        kwargs1 = {'opt_index_mat': global_index_mat, 'src_data': global_df}
        kwargs2 = {'opt_index_mat': solo_index_mat, 'src_data': solo_df}

        def highlight_cells(val, opt_index_mat, src_data):
            style_str = ';'.join([style_map[s] for s in styles])
            all_loc = np.where(val == src_data)
            loc = all_loc[0][:1], all_loc[1][:1]
            is_opt = np.all(opt_index_mat[loc])
            return style_str if is_opt else ''

        global_styled_df = global_df.style.map(highlight_cells, **kwargs1)
        solo_styled_df = solo_df.style.map(highlight_cells, **kwargs2)
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            global_styled_df.to_excel(writer, index=False, sheet_name='Global')
            solo_styled_df.to_excel(writer, index=False, sheet_name='Solo')

    def _get_table_df(
            self,
            algorithms: list[str],
            problem: str,
            np_per_dim: int,
            threshold_p: float):
        statistics = self._get_statistics(algorithms, problem, np_per_dim)
        keys = list(list(statistics.values())[0].keys())
        file_name = self._gen_name(algorithms[-1], problem, np_per_dim, keys, 'xlsx', single_file=True, in_log_scale=False)
        str_table, float_table = self._tabling(statistics, keys, threshold_p)
        global_index_mat, solo_index_mat = self._get_optimality_index_mat(float_table)

        global_counter = np.sum(global_index_mat, axis=0).reshape(1, -1)
        solo_counter = np.sum(solo_index_mat, axis=0).reshape(1, -1)

        df = pd.DataFrame(str_table, index=None)
        columns = copy.deepcopy(algorithms)
        columns.insert(0, "Clients")

        global_counter_df = pd.DataFrame(global_counter, columns=columns, index=None)
        solo_counter_df = pd.DataFrame(solo_counter, columns=columns, index=None)
        global_df = pd.concat([df, global_counter_df], ignore_index=True)
        solo_df = pd.concat([df, solo_counter_df], ignore_index=True)

        global_df.iloc[-1, 0] = 'Sum'
        solo_df.iloc[-1, 0] = 'Sum'
        return global_df, solo_df, global_index_mat, solo_index_mat, file_name

    def to_latex(
            self,
            algorithms: list[str],
            problem: str,
            np_per_dim: int,
            threshold_p: float = 0.05):
        tab_data = self._get_table_df(algorithms, problem, np_per_dim, threshold_p)
        data_df, _, hl_bool_mat, _, file_name = tab_data
        file_name = file_name.with_suffix('.tex')

        def highlight_cells(x):
            df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
            for i in range(hl_bool_mat.shape[0]):
                for j in range(hl_bool_mat.shape[1]):
                    if hl_bool_mat[i, j]:
                        df_styles.iloc[i, j] = 'background-color: gray'
            return df_styles

        styled_df = data_df.style.apply(highlight_cells, axis=None)
        latex_code = styled_df.to_latex(column_format='|'.join([''] + ['c'] * len(data_df.columns) + ['']),
                                        environment='longtable',
                                        caption=f"Table generated for {problem}, {np_per_dim} partitions",
                                        label=f"tab:{problem}_{np_per_dim}",
                                        position='htbp')

        with open(file_name, 'w') as f:
            f.write(latex_code)

    def to_console(
            self,
            algorithms: list[str],
            problem: str,
            np_per_dim: int,
            threshold_p: float=0.05):
        statistics = self._get_statistics(algorithms, problem, np_per_dim)
        keys = list(list(statistics.values())[0].keys())
        str_table, _ = self._tabling(statistics, keys, threshold_p)

        pd.set_option('display.colheader_justify', 'center')
        df = pd.DataFrame(str_table)
        print(f"Total {len(keys)} clients")
        print(df.to_string(index=False))

    @staticmethod
    def _get_optimality_index_mat(mean_dict):
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
            for j in range(col-1):
                if data_mat[i, j] < obj_val:
                    solo_index_mat[i, j] = True
        solo_index_mat = global_index_mat+solo_index_mat

        # row+1: the last row is the count of best solution of each algorithm
        # col+1: the first col is the index of Client
        add_row = np.zeros(shape=solo_index_mat.shape[1], dtype=bool)
        solo_index_mat = np.vstack((solo_index_mat, add_row))
        global_index_mat = np.vstack((global_index_mat, add_row))

        add_col = np.zeros(shape=solo_index_mat.shape[0], dtype=bool).reshape(-1, 1)
        solo_index_mat = np.hstack((add_col, solo_index_mat))
        global_index_mat = np.hstack((add_col, global_index_mat))
        return global_index_mat, solo_index_mat

    def _tabling(self, statistics: T_Sta, keys: list[str], threshold_p: float=0.05):
        algorithms = list(statistics.keys())
        obj_alg = algorithms[-1]
        obj_alg_merged = statistics[obj_alg]
        str_table = {"Clients":[i for i in keys]}
        float_table = {}
        for alg in algorithms[:-1]:
            str_res = []
            float_res = []
            for key in keys:
                opt_list1 = statistics[alg][key].opt_orig
                opt_list2 = obj_alg_merged[key].opt_orig
                mean1 = np.mean(opt_list1)
                mean2 = np.mean(opt_list2)
                suffix = self._get_t_test_suffix(opt_list1, opt_list2, mean1, mean2, threshold_p)
                str_res.append(f"{mean1:.2e}{suffix}")
                float_res.append(mean1)
            str_table.update({alg: str_res})
            float_table.update({alg: float_res})
        obj_alg_runs_opt = [obj_alg_merged[key].opt_orig for key in keys]
        obj_alg_opt_runs_mean = np.mean(obj_alg_runs_opt, axis=1)
        str_res = [f"{mean:.2e}" for mean in obj_alg_opt_runs_mean]
        str_table.update({obj_alg: str_res})
        float_table.update({obj_alg: obj_alg_opt_runs_mean.tolist()})

        return str_table, float_table

    @staticmethod
    def _get_t_test_suffix(opt_list1, opt_list2, mean1, mean2, threshold_p):
        diff = mean1 - mean2
        _, p = stats.ttest_ind(opt_list1, opt_list2)
        if p > threshold_p:
            suffix = "≈"
        elif diff > 0:
            suffix = "-"
        else:
            suffix = "+"
        return suffix

    @staticmethod
    def _update_ids(sorted_ids, chosen_ids):
        res_ids = []
        if not chosen_ids:
            return sorted_ids
        for cid in chosen_ids:
            if cid not in sorted_ids:
                logger.warning(f"Client {cid} is not in the result, please check your chosen client IDs")
                continue
            res_ids.append(cid)
        return res_ids

    @staticmethod
    def _plotting(ax, statistics: dict[str, dict[str: Statistics]], alg, key, showing_size, in_log_scale, alpha, color):
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
            return optimization_start_at
        else:
            return 0

    @staticmethod
    def _get_np_name(np_per_dim: int):
        return 'IID' if np_per_dim == 1 else f"NIID{np_per_dim}"

    def _gen_name(self, obj_alg, problem, np_per_dim, keys, suffix, single_file, in_log_scale):
        np_name = self._get_np_name(np_per_dim)
        log_tag = 'Logarithmic' if in_log_scale else 'Original'
        filedir = self._root / f"{time.strftime('%Y-%m-%d')}" / f"{obj_alg} / {problem}"
        if not filedir.exists():
            filedir.mkdir(parents=True)
        if single_file:
            file_name = filedir / f"{log_tag} {np_name}.{suffix}"
        else:
            file_name = {}
            for k in keys:
                file_name[k]=filedir/ 'One-by-one' / f"{k}.{suffix}"
        return file_name

    def _get_statistics(self, algorithms, problem, np_per_dim) -> T_Sta:
        statistics: T_Sta = {}
        for alg in algorithms:
            res = self._cache.get(f"{alg}_{problem}_{np_per_dim}")
            if not res:
                continue
            statis: dict[str, Statistics] = {}
            for k, v in res.items():
                curr_data = self._calculate_statistics(v['runs'])
                curr_data.fe_init = v['fe_init']
                curr_data.fe_max = v['fe_max']
                statis[k] = curr_data
            statistics[alg] = statis
        return statistics

    def _merge_one_algorithm(self, alg_name: str, problem: str, np_per_dim: int) -> bool:
        runs_path = self._get_runs_path(alg_name, problem, np_per_dim)
        if not runs_path:
            return False
        cache_key = f"{alg_name}_{problem}_{np_per_dim}"
        if cache_key in self._cache:
            return True

        runs_list = self._load_runs_data(runs_path)
        sorted_ids = self._check_attributes(runs_path, runs_list)
        runs_data: dict[str, dict] = {}
        for cid in sorted_ids:
            y_list, fe_init, fe_max = self._get_runs_y_data(cid, runs_list)
            runs_data[f"Client {cid:02d}"] = {
                'runs': y_list,
                'fe_init': fe_init,
                'fe_max': fe_max
            }
        self._cache[cache_key] = runs_data
        return True

    def _get_runs_path(self, alg_name: str, problem: str, np_per_dim: int):
        np_name = self._get_np_name(np_per_dim)
        res_dir = Path(f"{self._root}/{alg_name}/{problem}/{np_name}")
        if res_dir.exists():
            result_list = os.listdir(res_dir)
            result_list = [res_dir / f_name for f_name in result_list if f_name.endswith('.msgpack')]
            return result_list
        logger.warning(f"{res_dir} does not exist")

    @staticmethod
    def _load_runs_data(path_list: Union[list[Path], list[str]]) -> list[RunSolutions]:
        return [load_results(p) for p in path_list]

    def _check_attributes(self, path_list: list[Path], runs_list: list[RunSolutions]):
        ext_msg = "Data check failed, see out/logs/others.log for details"
        str_p_list = [str(os.path.join(*p.parts[-4:])) for p in path_list]

        # Check [Nun clients]
        num_clients = [[r.num_clients] for r in runs_list]
        col_title = ['num_clients']
        passed, msg = self._rows_check(num_clients, col_title, str_p_list, msg="Results attributes are not match")
        if not passed:
            logger.error(msg)
            exit(ext_msg)

        # Check [Client IDs]
        ids_list = [curr_run.sorted_ids for curr_run in runs_list]
        col_title = [f'Cid {i}' for i in ids_list[0]]
        passed, msg = self._rows_check(ids_list, col_title, str_p_list, msg="Client IDs are not match")
        if not passed:
            logger.error(msg)
            exit(ext_msg)
        else:
            sorted_ids = msg

        # Check [Solution size]
        size_of_clients = []
        for run_res in runs_list:
            curr_c_size = [run_res.get_solutions(cid).size for cid in sorted_ids]
            size_of_clients.append(curr_c_size)
        passed, msg = self._rows_check(size_of_clients, col_title, str_p_list, msg="Solution size are not match")
        if not passed:
            logger.error(msg)
            exit(ext_msg)

        return sorted_ids

    @staticmethod
    def _rows_check(data_list: list[list], col_title: list[str], row_title: list[str], msg=''):
        try:
            data_mat = np.array(data_list, dtype=int)
            if not np.all(np.equal(data_mat, data_mat[0])):
                tab = tabulate(data_list, headers=col_title, showindex=row_title, tablefmt='psql')
                return False, f"{'=' * 10} {msg} {'=' * 10}\n{tab}"
        except Exception:
            tab = tabulate(data_list, headers=col_title, showindex=row_title, tablefmt='psql')
            return False, f"{'=' * 10} {msg} {'=' * 10}\n{tab}"

        return True, data_mat[0]

    @staticmethod
    def _get_runs_y_data(cid, runs_list):
        curr_client_runs_data = []
        fe_init = 0
        fe_max = 0
        for run_res in runs_list:
            solution = run_res.get_solutions(cid)
            fe_init = solution.fe_init
            fe_max = solution.size
            curr_client_runs_data.append(solution.y_homo_decrease)
        return curr_client_runs_data, fe_init, fe_max

    @staticmethod
    def _calculate_statistics(data: Union[list, ndarray]) -> Statistics:
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
    def _find_grid_shape(size):
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
        a = math.ceil(math.sqrt(size))
        b = math.ceil(size / a)

        while a * b > size:
            if a > b:
                a -= 1
            else:
                b -= 1

        while a * b < size:
            if a < b:
                a += 1
            else:
                b += 1

        if a > b and (a - 1) * (b + 1) >= size:
            a -= 1
            b += 1
        elif b > a and (a + 1) * (b - 1) >= size:
            a += 1
            b -= 1

        return (a, b) if a > b else (b, a)

    @staticmethod
    def _aspect_ratio(width, height, scale):
        return width * scale, height * scale

    def show_raw_results(self, alg_name: str, problem: str, np_per_dim: int):
        result_list = self._get_runs_path(alg_name, problem, np_per_dim)
        result_list = [str(path) for path in result_list]
        if result_list:
            print(f"{'=' * 10} {alg_name}+{problem}+np({np_per_dim}) total {len(result_list)} files {'=' * 10}")
            print('\n'.join(sorted(result_list)))


class AnalyzeResults:
    def __init__(self):
        try:
            settings = load_analyses_settings()
            algorithms = settings.get('algorithms')
            problems = settings.get('problems')
            np_per_dim = settings.get('np_per_dim')
            prod = itertools.product(algorithms, problems, np_per_dim)
            algors = list(set(sum(algorithms, [])))
            self.combinations = list(prod)
            self.analyzer = Analyzer(settings.get('results'))
            self.analyzer.init_data(algors, problems, np_per_dim)
            self.analyzer_available = True
        except FileNotFoundError:
            self.analyzer_available = False

    def show_combinations(self):
        header = ['algorithms in comparison', 'on problem', 'partitions']
        tab = tabulate(self.combinations, headers=header, tablefmt='rounded_grid')
        clear_console()
        print(tab)

    def show_raw_results(self):
        done = set()
        for comb in self.combinations:
            for alg in comb[0]:
                if alg not in done:
                    self.analyzer.show_raw_results(alg, *comb[1:])
                    done.add(f"{alg}{comb[1]}{comb[2]}")
                else:
                    pass

    def to_curve(self, **kwargs):
        for comb in tqdm(self.combinations, desc='Saving', unit='Img', ncols=100):
            self.analyzer.to_curve(*comb, **kwargs)

    def to_excel(self, **kwargs):
        for comb in self.combinations:
            self.analyzer.to_excel(*comb, **kwargs)

    def to_latex(self, **kwargs):
        for comb in self.combinations:
            self.analyzer.to_latex(*comb, **kwargs)

    def to_console(self, **kwargs):
        for comb in self.combinations:
            self.analyzer.to_console(*comb, **kwargs)
