import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # Do not remove this import
import seaborn
import time
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import validate_call, Field
from pyfmto.utilities import SeabornPalettes, load_yaml, logger
from tqdm import tqdm
from typing import Optional, Literal, Annotated, final

from .utils import ReporterUtils, MergedResults, MetaData

_ = scienceplots.stylesheets  # This is to suppress the 'unused import' warning
T_Suffix = Literal['.png', '.jpg', '.eps', '.svg', '.pdf']
T_Fraction = Annotated[float, Field(ge=0., le=1.)]
T_Levels10 = Annotated[int, Field(ge=1, le=10)]

__all__ = ['Reports']


class ReportGenerator(ABC):

    def __init__(self, data_size_req: int):
        self.data_size_req = data_size_req
        self.utils = ReporterUtils

    @final
    def _check_data(self, data: MetaData):
        if len(data) >= self.data_size_req:
            return True
        else:
            print(f"Not enough data for {self.__class__.__name__}")
            return False

    @abstractmethod
    def _generate(self, *args, **kwargs):
        pass

    @final
    def generate(self, data: MetaData, *args, **kwargs):
        self._check_data(data)
        self._generate(data, *args, **kwargs)


class CurveGenerator(ReportGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            figsize: tuple[float, float, float],
            alpha: float,
            palette: SeabornPalettes,
            suffix: str,
            styles: tuple[str, ...],
            showing_size: Optional[int],
            quality: int,
            merge: bool,
            clear: bool,
            on_log_scale: bool,
    ):
        # Prepare the data
        w, h, s = figsize
        _figsize = w * s, h * s
        _quality = {'dpi': 100 * quality}
        _suffix = self.utils.check_suffix(suffix, merge)
        log_tag = ' log' if on_log_scale else ''
        # Plot the data
        colors = seaborn.color_palette(str(palette), data.alg_num - 1).as_hex()
        colors.append("#ff0000")
        with plt.style.context(styles):
            filedir = data.filedir.parent / f"{data.filedir.name} curve{log_tag}"
            filedir.mkdir(parents=True, exist_ok=True)
            for c_name in data.clt_names:
                plt.figure(figsize=_figsize, **_quality)
                for (alg_name, merged_data), color in zip(data.items(), colors):
                    start_idx = self.utils.plotting(
                        plt, merged_data, alg_name, c_name, showing_size, on_log_scale, alpha, color
                    )
                    plt.title('Client')
                    plt.xlabel('Iteration')
                    plt.ylabel('Fitness')
                if start_idx > 0:
                    plt.axvline(x=start_idx, color='gray', linestyle='--')
                plt.legend()
                plt.tight_layout()
                plt.savefig(filedir / f'{c_name}{_suffix}')
                plt.close()

            if merge:
                self.utils.merge_images_in(filedir, clear)


class ViolinGenerator(ReportGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            suffix: str,
            figsize: tuple[float, float, float],
            quality: int,
            merge: bool,
            clear: bool
    ):
        filedir = data.filedir.parent / f"{data.filedir.name} violin"
        filedir.mkdir(parents=True, exist_ok=True)
        w, h, s = figsize
        _suffix = self.utils.check_suffix(suffix, merge)
        _figsize = w * s, h * s
        _quality = 100. * quality
        alg_res: MergedResults = data[data.alg_names[-1]]
        for clt_name, sta in alg_res.items():
            title = f"{clt_name} of {data.alg_names[-1]} on {data.problem}"
            self.utils.plot_violin(sta, _figsize, filedir / f"{clt_name}{_suffix}", title=title, dpi=_quality)
        if merge:
            self.utils.merge_images_in(filedir, clear)


class TableGenerator(ReportGenerator, ABC):

    def _get_table_data(
            self,
            data: MetaData,
            pvalue: float
    ):
        str_table, float_table = self._tabling(data, pvalue)
        global_index_mat, solo_index_mat = self.utils.get_optimality_index_mat(float_table)

        global_counter = np.sum(global_index_mat, axis=0).reshape(1, -1)
        solo_counter = np.sum(solo_index_mat, axis=0).reshape(1, -1)

        df = pd.DataFrame(str_table, index=None)
        columns = data.alg_names
        columns.insert(0, "Clients")

        global_counter_df = pd.DataFrame(global_counter, columns=columns, index=None)
        solo_counter_df = pd.DataFrame(solo_counter, columns=columns, index=None)
        global_df = pd.concat([df, global_counter_df], ignore_index=True)
        solo_df = pd.concat([df, solo_counter_df], ignore_index=True)

        global_df.iloc[-1, 0] = 'Sum'
        solo_df.iloc[-1, 0] = 'Sum'
        return global_df, solo_df, global_index_mat, solo_index_mat

    def _tabling(
            self,
            data: MetaData,
            pvalue: float
    ):
        obj_alg = data.alg_names[-1]
        obj_alg_merged = data[obj_alg]
        str_table = {"Clients": data.clt_names}
        float_table = {}
        for alg in data.alg_names[:-1]:
            str_res = []
            float_res = []
            for c_name in data.clt_names:
                c_data = data[alg].get_statis(c_name)
                c_data_obj = obj_alg_merged.get_statis(c_name)
                opt_list1 = c_data.y_dec_statis.opt
                opt_list2 = c_data_obj.y_dec_statis.opt
                mean1 = np.mean(opt_list1)
                mean2 = np.mean(opt_list2)
                suffix = self.utils.get_t_test_suffix(opt_list1, opt_list2, mean1, mean2, pvalue)
                str_res.append(f"{mean1:.2e}{suffix}")
                float_res.append(mean1)
            str_table.update({alg: str_res})
            float_table.update({alg: float_res})
        obj_alg_runs_opt = [obj_alg_merged.get_statis(c_name).y_dec_statis.opt for c_name in data.clt_names]
        obj_alg_opt_runs_mean = np.mean(obj_alg_runs_opt, axis=1)
        str_res = [f"{mean:.2e}" for mean in obj_alg_opt_runs_mean]
        str_table.update({obj_alg: str_res})
        float_table.update({obj_alg: obj_alg_opt_runs_mean.tolist()})
        return str_table, float_table


class ExcelGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: float,
            styles: tuple[str, ...]
    ):
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
        df_data = self._get_table_data(data, pvalue)
        global_df, solo_df, global_index_mat, solo_index_mat = df_data
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

        with pd.ExcelWriter(data.filedir.with_suffix('.xlsx'), engine='openpyxl') as writer:
            global_styled_df.to_excel(writer, index=False, sheet_name='Global')
            solo_styled_df.to_excel(writer, index=False, sheet_name='Solo')


class ConsoleGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: float,
            **kwargs
    ):
        str_table, _ = self._tabling(data, pvalue)
        pd.set_option('display.colheader_justify', 'center')
        df = pd.DataFrame(str_table)
        print(f"Total {data.clt_num} clients")
        print(df.to_string(index=False))


class LatexGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: float
    ):
        tab_data = self._get_table_data(data, pvalue)
        data_df, _, hl_bool_mat, _ = tab_data

        def highlight_cells(x):
            df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
            for i in range(hl_bool_mat.shape[0]):
                for j in range(hl_bool_mat.shape[1]):
                    if hl_bool_mat[i, j]:
                        df_styles.iloc[i, j] = 'background-color: gray'
            return df_styles

        styled_df = data_df.style.apply(highlight_cells, axis=None)
        latex_code = styled_df.to_latex(column_format='c' * len(data_df.columns),
                                        environment='table',
                                        caption=f"Table generated for {data.problem}, {data.npd_name}",
                                        label=f"tab:{data.problem}_{data.npd_name}",
                                        position_float='centering',
                                        hrules=True,
                                        position='htbp')

        latex_code = latex_code.replace('background-colorgray', 'cellcolor{gray!30}')
        latex_code = latex_code.replace('≈', r'$\approx$')
        latex_code = latex_code.replace('_', r'\_')
        with open(data.filedir.with_suffix('.txt'), 'w') as f:
            f.write(latex_code)


class Reporter:
    def __init__(self, results, initialize_comb):
        self._root = Path(results)
        self._cache: dict[str, MergedResults] = {}
        self._generators: dict[str, ReportGenerator] = {}  # Registry of report generators
        self._load_data(initialize_comb)

    def _load_data(self, combinations):
        for comb in combinations:
            subdir = '/'.join(comb)
            if subdir not in self._cache:
                runs_data = ReporterUtils.load_runs_data(self._root / subdir)
                if runs_data:
                    self._cache[subdir] = MergedResults(runs_data)

    def register_generator(self, name: str, generator: ReportGenerator):
        self._generators[name] = generator

    def generate_report(self, generator_name: str, algorithms: list[str], problem: str, npd_name: str, **kwargs):
        if not generator_name not in self._generators:
            raise ValueError(f"No generator registered for {generator_name}")
        data = self._prepare_data(algorithms, problem, npd_name)
        return self._generators[generator_name].generate(data, **kwargs)

    def _prepare_data(self, algorithms: list[str], problem: str, npd_name: str) -> MetaData:
        data: dict[str, MergedResults] = {}
        for algorithm in algorithms:
            cache_key = f"{algorithm}/{problem}/{npd_name}"
            merged_res = self._cache.get(cache_key)
            if merged_res:
                data[algorithm] = merged_res
        filedir = self._get_output_dir(list(data.keys()), problem, npd_name)
        return MetaData(data, problem, npd_name, filedir)

    def _get_output_dir(self, algorithms: list[str], problem: str, npd_name: str) -> Path:
        filedir = self._root / f"{time.strftime('%Y-%m-%d')}" / f"{algorithms[-1]}" / f"{problem}"
        filedir.mkdir(parents=True, exist_ok=True)
        file_name = filedir / f"{npd_name}"
        return file_name


class Reports:
    def __init__(self):
        all_conf = load_yaml('config.yaml')
        settings = ReporterUtils.parse_reporter_config(all_conf.get('reporter'), all_conf.get('problems', {}))
        self.combinations = settings.pop('analysis_comb')
        self.reporter = Reporter(**settings)

        self.reporter.register_generator('to_curve', CurveGenerator(data_size_req=2))
        self.reporter.register_generator('to_excel', ExcelGenerator(data_size_req=2))
        self.reporter.register_generator('to_violin', ViolinGenerator(data_size_req=1))
        self.reporter.register_generator('to_console', ConsoleGenerator(data_size_req=2))
        self.reporter.register_generator('to_latex', LatexGenerator(data_size_req=2))

    @validate_call
    def to_curve(
            self,
            *,
            figsize: tuple[float, float, float] = (3., 2.3, 1.),
            alpha: T_Fraction = .2,
            palette: SeabornPalettes = SeabornPalettes.bright,
            suffix: T_Suffix = '.png',
            styles: tuple[str, ...] = ('science', 'ieee', 'no-latex'),
            showing_size: int = -1,
            quality: T_Levels10 = 3,
            merge: bool = True,
            clear: bool = True,
            on_log_scale: bool = False
    ) -> None:
        """
        Generate and save plots of performance curves for specified algorithms and problem settings.

        Parameters
        ----------
        figsize : tuple[float, float, float]
            Controlling the width-to-height ratio and the scale of the ratio. The third value is the scaling factor.
        alpha : float, optional
            Transparency of the Standard Error region, ranging from 0 (completely transparent) to 1 (completely opaque).
        palette :
            The palette argument in `seaborn.plotviolin`, the `pyfmto.utilities.SeabornPalette` class can help you try
            different options easier.
        suffix : str, optional
            File format suffix for the output image. Supported formats include 'png', 'jpg', 'eps', 'svg', 'pdf'.
        styles : Union[list[str], tuple[str]], optional
            SciencePlots style parameters. Refer to the SciencePlots documentation for available styles.
        showing_size : int, optional
            Number of data points to use for plotting, taken from the last `showing_size` iterations of the convergence
            sequence.
            If None, use `6 * dim`.
        quality : Optional[int], optional
            Image quality parameter, affecting the quality of scalar images. Valid values are integers from 1 to 9.
        merge : bool, optional
            If True, all curves are plotted on a single figure. If False, each client's curves are plotted in separate
            figures.
        clear: bool, optional
            If True, clear plots output in one_by_one
        on_log_scale : bool, optional
            If True, the plot is generated on a logarithmic scale. If False, the plot is generated on an original scale.
        """
        need_new_line = True
        for comb in tqdm(self.combinations, desc='Saving', unit='Img', ncols=100):
            try:
                self.reporter.generate_report(
                    'to_curve',
                    *comb,
                    figsize=figsize,
                    alpha=alpha,
                    palette=palette,
                    suffix=suffix,
                    styles=styles,
                    showing_size=showing_size,
                    quality=quality,
                    merge=merge,
                    clear=clear,
                    on_log_scale=on_log_scale,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                if need_new_line:
                    print('\n')
                    need_new_line = False
                print(e)

    @validate_call
    def to_excel(
            self,
            *,
            pvalue: T_Fraction = 0.05,
            styles: tuple[str, ...] = ('color-bg-grey', 'style-font-bold', 'style-font-underline')
    ) -> None:
        """
        Generate and save an Excel file containing performance statistics for specified algorithms and problem settings.

        Parameters
        ----------
        pvalue : float, optional
            T-test threshold parameter to determine statistical significance. Default is 0.05.
        styles : Union[list[str], tuple[str]], optional
            List or tuple of style parameters to apply to the Excel cells.

        Notes
        -----
            The following styles are supported for personalized excel style:
             - ``color-bg-[red|grey|green|blue|yellow|purple|orange|pink]`` ---Background colors (only one can be
             applied, supported colors are in [])
             - ``color-font-[red|green|blue|yellow|purple|orange|pink]`` ---Font colors (only one can be applied,
             supported colors are in [])
             - ``type-font-[bold|italic|underline]`` ---Font types (multiple can be applied, supported types are in [])
        """
        for comb in self.combinations:
            try:
                self.reporter.generate_report(
                    'to_excel',
                    *comb,
                    pvalue=pvalue,
                    styles=styles,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                print(e)

    @validate_call
    def to_latex(
            self,
            *,
            pvalue: T_Fraction = 0.05
    ) -> None:
        """
        Generate and save an .tex file containing performance statistics for specified algorithms and problem settings.

        Parameters
        ----------
        pvalue : float, optional
            T-test threshold parameter to determine statistical significance. Default is 0.05.
        """
        for comb in self.combinations:
            try:
                self.reporter.generate_report('to_latex', *comb, pvalue=pvalue)
            except Exception as e:
                logger.error(traceback.format_exc())
                print(e)

    @validate_call
    def to_console(
            self,
            *,
            pvalue: float = 0.05,
    ) -> None:
        """
        Print the performance statistics table of specified algorithms and problem settings to the console.

        Parameters
        ----------
        pvalue : float, optional
            T-test threshold for determining statistical significance. Default is 0.05.
        """
        for comb in self.combinations:
            try:
                self.reporter.generate_report('to_console', *comb, pvalue=pvalue)
            except Exception as e:
                logger.error(traceback.format_exc())
                print(e)

    @validate_call
    def to_violin(
            self,
            *,
            suffix: T_Suffix = '.png',
            figsize: tuple[float, float, float] = (5., 3., 1.),
            quality: T_Levels10 = 3,
            merge: bool = True,
            clear: bool = True
    ) -> None:
        """
        Parameters
        ----------
        suffix: str
            image suffix, such as png, pdf, svg
        figsize: tuple
            Controlling the (width, height, scale). the figsize is calculated by (width*scale, height*scale)
        quality : Optional[int], optional
            Image quality parameter, affecting the quality of scalar images. Valid values are integers from 1 to 9
        merge: bool
            if true, merge all separate images into a single image, and the suffix will be fixed to PNG
        clear: bool
            only takes effect when merge is true

        Returns
        -------
            None
        """
        for comb in self.combinations:
            try:
                self.reporter.generate_report(
                    'to_violin',
                    *comb,
                    suffix=suffix,
                    figsize=figsize,
                    merge=merge,
                    clear=clear,
                    quality=quality,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                print(e)
