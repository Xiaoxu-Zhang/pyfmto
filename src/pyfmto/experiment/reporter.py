import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from typing import Annotated, Literal, final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # Do not remove this import
import seaborn
from pydantic import Field, validate_call
from rich.progress import track

from ..core.typing import PaletteOptions
from ..utilities.loggers import clogger
from .config import ReporterConfig
from .utils import FILL, INDENT, SLEEP_TIME, MergedResults, MetaData, ReporterUtils, min_time_lapse

_ = scienceplots.stylesheets  # This is to suppress the 'unused import' warning
T_Suffix = Literal['.png', '.jpg', '.jpeg', '.svg', '.pdf']
T_Fraction = Annotated[float, Field(ge=0., le=1.)]
T_Levels10 = Annotated[int, Field(ge=1, le=10)]

__all__ = [
    'ConsoleGenerator',
    'CurveGenerator',
    'ExcelGenerator',
    'LatexGenerator',
    'ReportGenerator',
    'Reporter',
    'TableGenerator',
    'ViolinGenerator',
]


class ReportGenerator(ABC):
    data_size_req: int

    def __init__(self):
        self.utils = ReporterUtils

    @final
    def _check_data(self, data: MetaData):
        time.sleep(SLEEP_TIME)
        clogger.debug(f"{INDENT}Checking data size...")
        if len(data) >= self.data_size_req:
            return True
        else:
            time.sleep(SLEEP_TIME)
            clogger.warn(
                f"{self.__class__.__name__} require at least {self.data_size_req} "
                f"data to generate report, got {len(data)} instead.")

    @abstractmethod
    def _generate(self, *args, **kwargs):
        raise NotImplementedError  # pragma: no cover

    @final
    @min_time_lapse(0.5)
    def generate(self, data: MetaData, *args, **kwargs):
        self._check_data(data)
        self._generate(data, *args, **kwargs)


class CurveGenerator(ReportGenerator):
    data_size_req = 2

    def _generate(
            self,
            data: MetaData,
            *,
            figsize: tuple[float, float, float] = (3., 2.3, 1.),
            alpha: T_Fraction = .2,
            palette: PaletteOptions = 'bright',
            suffix: T_Suffix = '.png',
            styles: tuple[str, ...] = ('science', 'ieee', 'no-latex'),
            showing_size: int = -1,
            quality: T_Levels10 = 3,
            merge: bool = True,
            clear: bool = True,
            on_log_scale: bool = False
    ):
        # Prepare the data
        w, h, s = figsize
        _figsize = w * s, h * s
        _quality = {'dpi': 100 * quality}
        _suffix = self.utils.check_suffix(suffix, merge)
        log_tag = '-log' if on_log_scale else ''
        # Plot the data
        colors = seaborn.color_palette(palette, data.alg_num - 1).as_hex()
        colors.append("#ff0000")
        with plt.style.context(styles):
            filedir = data.report_filename.parent / f"{data.report_filename.name}-curve{log_tag}"
            filedir.mkdir(parents=True, exist_ok=True)
            time.sleep(SLEEP_TIME)
            clogger.debug(f"{INDENT}Generating single curve...")
            for c_name in track(data.clt_names, description="SubImg", transient=True):
                plt.figure(figsize=_figsize, **_quality)
                for (alg_name, merged_data), color in zip(data.items(), colors):
                    start_idx = self.utils.plotting(
                        plt, merged_data, alg_name, c_name, showing_size, on_log_scale, alpha, color
                    )
                    plt.title(c_name)
                    plt.xlabel('Iteration')
                    plt.ylabel('Fitness')
                if start_idx > 0:
                    plt.axvline(x=start_idx, color='gray', linestyle='--')
                plt.legend()
                plt.tight_layout()
                plt.savefig(filedir / f'{c_name}{_suffix}')
                plt.close()

            if merge:
                clogger.debug(f"{INDENT}Merging images...")
                self.utils.merge_images_in(filedir, clear)


class ViolinGenerator(ReportGenerator):
    data_size_req = 1

    def _generate(
            self,
            data: MetaData,
            *,
            suffix: T_Suffix = '.png',
            figsize: tuple[float, float, float] = (5., 3., 1.),
            quality: T_Levels10 = 3,
            merge: bool = True,
            clear: bool = True
    ):
        filedir = data.report_filename.parent / f"{data.report_filename.name} violin"
        filedir.mkdir(parents=True, exist_ok=True)
        w, h, s = figsize
        _suffix = self.utils.check_suffix(suffix, merge)
        _figsize = w * s, h * s
        _quality = 100. * quality
        alg_res: MergedResults = data[data.alg_names[-1]]
        for clt_name, sta in track(alg_res.items(), description="SubImg", transient=True):
            title = f"{clt_name} of {data.alg_names[-1]} on {data.problem}"
            self.utils.plot_violin(sta, _figsize, filedir / f"{clt_name}{_suffix}", title=title, dpi=_quality)
        if merge:
            self.utils.merge_images_in(filedir, clear)


class TableGenerator(ReportGenerator, ABC):
    data_size_req = 2

    def _get_table_data(
            self,
            data: MetaData,
            pvalue: float
    ):
        str_table, float_table, counter = self._tabling(data, pvalue)
        mask = self.utils.get_optimality_mask_mat(float_table)

        # global_counter = np.sum(mask, axis=0).reshape(1, -1)

        df = pd.DataFrame(str_table, index=None)
        columns = data.alg_names
        columns.insert(0, "Clients")

        global_counter_df = pd.DataFrame(counter, columns=columns, index=None)
        global_df = pd.concat([df, global_counter_df], ignore_index=True)

        # global_df.iloc[-1, 0] = 'Sum'
        return global_df, mask

    def _tabling(
            self,
            data: MetaData,
            pvalue: float
    ):
        obj_alg = data.alg_names[-1]
        obj_alg_merged = data[obj_alg]
        str_table = {"Clients": data.clt_names}
        float_table = {}
        counter: list[str] = ['+/-/≈']
        for alg in data.alg_names[:-1]:
            str_res = []
            float_res = []
            _cnt = {'+': 0, '-': 0, '≈': 0}
            for c_name in data.clt_names:
                c_data = data[alg].get_statis(c_name)
                c_data_obj = obj_alg_merged.get_statis(c_name)
                opt_list1 = c_data.y_dec_statis.opt
                opt_list2 = c_data_obj.y_dec_statis.opt
                mean1 = np.mean(opt_list1)
                mean2 = np.mean(opt_list2)
                suffix = self.utils.get_t_test_suffix(opt_list1, opt_list2, mean1, mean2, pvalue)
                str_res.append(f"{mean1:.2e}{suffix}")
                _cnt[suffix] += 1
                float_res.append(mean1)
            counter.append(f"{_cnt['+']}/{_cnt['-']}/{_cnt['≈']}")
            str_table.update({alg: str_res})
            float_table.update({alg: float_res})
        counter.append('')
        obj_alg_runs_opt = [obj_alg_merged.get_statis(c_name).y_dec_statis.opt for c_name in data.clt_names]
        obj_alg_opt_runs_mean = np.mean(obj_alg_runs_opt, axis=1)
        str_res = [f"{mean:.2e}" for mean in obj_alg_opt_runs_mean]
        str_table.update({obj_alg: str_res})
        float_table.update({obj_alg: obj_alg_opt_runs_mean.tolist()})
        return str_table, float_table, [counter]


class ExcelGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: T_Fraction = 0.05,
            styles: tuple[str, ...] = ('color-bg-grey', 'style-font-bold', 'style-font-underline')
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
        df, mask = self._get_table_data(data, pvalue)
        kwargs1 = {'opt_index_mat': mask, 'src_data': df}

        def highlight_cells(val, opt_index_mat, src_data):
            style_str = ';'.join([style_map[s] for s in styles])
            all_loc = np.where(val == src_data)
            loc = all_loc[0][:1], all_loc[1][:1]
            is_opt = np.all(opt_index_mat[loc])
            return style_str if is_opt else ''

        styled_df = df.style.map(highlight_cells, **kwargs1)

        writer: pd.ExcelWriter
        with pd.ExcelWriter(data.report_filename.with_suffix('.xlsx'), engine='openpyxl') as writer:
            styled_df.to_excel(writer, index=False, sheet_name='Global')


class ConsoleGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: T_Fraction = 0.05,
    ):
        str_table, _, _ = self._tabling(data, pvalue)
        pd.set_option('display.colheader_justify', 'center')
        df = pd.DataFrame(str_table)
        clogger.debug(f"{INDENT}Total {data.clt_num} clients")
        print(df.to_string(index=False))


class LatexGenerator(TableGenerator):

    def _generate(
            self,
            data: MetaData,
            *,
            pvalue: T_Fraction = 0.05
    ):
        df, mask = self._get_table_data(data, pvalue)

        def highlight_cells(x):
            df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j]:
                        df_styles.iloc[i, j] = 'background-color: gray'
            return df_styles

        styled_df = df.style.apply(highlight_cells, axis=None)
        alg_str = ','.join(data.alg_names[:-1]) + ' and ' + data.alg_names[-1]
        caption = (f"The average/mean best optimum obtained by {alg_str}, {len(data.alg_names)} "
                   f"algorithms under comparison, in handling {data.dim}-dimensional MaTOP problems with "
                   f"$NPD={data.npd}$ over {data.num_runs} independent runs.")
        latex_code = styled_df.to_latex(column_format='c' * len(df.columns),
                                        environment='table',
                                        caption=caption,
                                        label=f"tab:{data.problem}_{data.npd}",
                                        position_float='centering',
                                        hrules=True,
                                        position='htbp')

        latex_code = latex_code.replace('background-colorgray', 'cellcolor{gray!30}')
        latex_code = latex_code.replace('≈', r'$\approx$')
        latex_code = latex_code.replace('_', r'\_')
        with open(data.report_filename.with_suffix('.txt'), 'w') as f:
            f.write(latex_code)


class GeneratorManager:
    def __init__(self, conf: ReporterConfig):
        self.conf = conf
        self._cache: dict[str, MergedResults] = {}
        self._generators: dict[str, ReportGenerator] = {}  # Registry of report generators
        self._load_data()

    def _load_data(self):
        clogger.debug(f"{'Loading data':=^5}")
        for exp in self.conf.experiments:
            alg, prob, npd = exp.result_dir.parts[-3:]
            clogger.debug(f"{INDENT}Loading {alg}-{prob}({npd})")
            key = f"{alg}/{prob}/{npd}"
            if key not in self._cache:
                runs_data = ReporterUtils.load_runs_data(exp.result_dir, prefix=exp.prefix)
                if runs_data:
                    self._cache[key] = MergedResults(runs_data)
                else:
                    clogger.warning(
                        textwrap.indent("\nResult file not found:\n"
                        f"  CacheKey: {key}\n"
                        f"  FileRoot: {exp.result_dir}\n"
                        f"  NameRule: {exp.prefix}[{'Any'}].msgpack\n", INDENT)
                    )

    def register_generator(self, name: str, generator: ReportGenerator):
        self._generators[name] = generator

    def generate_report(self, generator_name: str, algorithms: list[str], problem: str, npd_name: str, **kwargs):
        alg_str = ', '.join(algorithms)
        msg = f"""
          Algs: [green]{alg_str}[/green]
          Prob: [green]{problem}({npd_name})[/green]
        """
        time.sleep(0.2)
        clogger.debug(textwrap.dedent(msg))
        if generator_name not in self._generators:
            clogger.warn(f"{INDENT}No generator registered for {generator_name}")
            return None
        try:
            data = self._prepare_data(algorithms, problem, npd_name)
            return self._generators[generator_name].generate(data, **kwargs)
        except Exception:
            clogger.error(traceback.format_exc())

    @min_time_lapse(0.5)
    def _prepare_data(self, algorithms: list[str], problem: str, npd_name: str) -> MetaData:
        time.sleep(SLEEP_TIME)
        clogger.debug(f"{INDENT}Preparing data...")
        data: dict[str, MergedResults] = {}
        for algorithm in algorithms:
            cache_key = f"{algorithm}/{problem}/{npd_name}"
            merged_res = self._cache.get(cache_key)
            if merged_res:
                data[algorithm] = merged_res
        str_bad = ', '.join(set(algorithms) - set(data.keys()))
        str_good = ', '.join(list(data.keys()))
        time.sleep(SLEEP_TIME)
        clogger.info(f"{INDENT}Available data: [green]{str_good}[/green]")
        if str_bad:
            time.sleep(SLEEP_TIME)
            clogger.warn(f"{INDENT}Unavailable data: [red]{str_bad}[/red]")
        return MetaData(data, problem, npd_name, self.conf.results)


class Reporter:
    def __init__(self, conf: ReporterConfig):
        self.conf = conf
        self.manager = GeneratorManager(self.conf)
        self.manager.register_generator('curve', CurveGenerator())
        self.manager.register_generator('excel', ExcelGenerator())
        self.manager.register_generator('violin', ViolinGenerator())
        self.manager.register_generator('console', ConsoleGenerator())
        self.manager.register_generator('latex', LatexGenerator())

    def report(self):
        if not self.conf.formats:
            clogger.warn(f"{'No formats specified':=^5}")
            return
        for fmt in self.conf.formats:
            if hasattr(self, f'to_{fmt}'):
                getattr(self, f'to_{fmt}')(**self.conf.params.get(fmt, {}))
            else:
                clogger.warn(f"Format [red]{fmt}[/red] is invalid")

    @validate_call
    def to_curve(
            self,
            *,
            figsize: tuple[float, float, float] = (3., 2.3, 1.),
            alpha: T_Fraction = .2,
            palette: PaletteOptions = 'bright',
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
            The palette argument in `seaborn.plotviolin`
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
        clogger.debug(f"{FILL}Generating [green bold]Curve[/green bold]{FILL}")
        for group in track(self.conf.groups, description='Curve', transient=True):
            self.manager.generate_report(
                'curve',
                *group,
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
        clogger.debug(f"{FILL}Generating [green bold]Excel[/green bold]{FILL}")
        for group in track(self.conf.groups, description='Excel', transient=True):
            self.manager.generate_report(
                'excel',
                *group,
                pvalue=pvalue,
                styles=styles,
            )
            time.sleep(0.5)

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
        clogger.debug(f"{FILL}Generating [green bold]LaTeX[/green bold]{FILL}")
        for group in track(self.conf.groups, description='LaTeX', transient=True):
            self.manager.generate_report(
                'latex',
                *group,
                pvalue=pvalue
            )
            time.sleep(0.5)

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
        clogger.debug(f"{FILL}Printing to [green bold]Console[/green bold]{FILL}")
        for comb in track(self.conf.groups, description='Console', transient=True):
            self.manager.generate_report(
                'console',
                *comb,
                pvalue=pvalue
            )

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
        clogger.debug(f"{FILL}Generating [green bold]Violin[/green bold]{FILL}")
        for group in track(self.conf.groups, description='Violin', transient=True):
            self.manager.generate_report(
                'violin',
                *group,
                suffix=suffix,
                figsize=figsize,
                merge=merge,
                clear=clear,
                quality=quality,
            )
