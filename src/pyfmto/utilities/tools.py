import os
import platform
import subprocess

from contextlib import contextmanager
from rich.console import Console
from rich.table import Table, box
from tabulate import tabulate
from typing import Literal
from .loggers import logger

__all__ = [
    'colored',
    'clear_console',
    'titled_tabulate',
    'tabulate_formats',
    'matched_str_head',
    'terminate_popen',
    'redirect_warnings',
    'print_dict_as_table'
]


class TabulatesFormats:
    plain = 'plain'
    simple = 'simple'
    grid = 'grid'
    simple_grid = 'simple_grid'
    rounded_grid = 'rounded_grid'
    heavy_grid = 'heavy_grid'
    mixed_grid = 'mixed_grid'
    double_grid = 'double_grid'
    fancy_grid = 'fancy_grid'
    outline = 'outline'
    simple_outline = 'simple_outline'
    rounded_outline = 'rounded_outline'
    mixed_outline = 'mixed_outline'
    double_outline = 'double_outline'
    fancy_outline = 'fancy_outline'
    pipe = 'pipe'
    presto = 'presto'
    orgtbl = 'orgtbl'
    rst = 'rst'
    mediawiki = 'mediawiki'
    html = 'html'
    latex = 'latex'
    latex_raw = 'latex_raw'
    latex_booktabs = 'latex_booktabs'
    latex_longtable = 'latex_longtable'


tabulate_formats = TabulatesFormats()


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


def print_dict_as_table(data: dict[str, list], true_color="green", false_color="red", title=None):
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"all values must have the same length: {lengths}")
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    for key in data.keys():
        table.add_column(str(key), max_width=50, justify="center", overflow='fold', vertical='middle')
    num_rows = lengths.pop()
    keys = list(data.keys())
    for i in range(num_rows):
        row = []
        for key in keys:
            val = data[key][i]
            if isinstance(val, bool):
                color = true_color if val else false_color
                row.append(f"[{color}]{val}[/{color}]")
            else:
                row.append(str(val))
        table.add_row(*row)
    console = Console()
    console.print(table)


def colored(text: str, color: Literal['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'reset']):
    color_map = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    if color not in color_map:
        raise ValueError(f"Unsupported color: {color}")

    return f"{color_map[color]}{text}{color_map['reset']}"


def titled_tabulate(title: str, fill_char: str, *args, **kwargs):
    title = ' ' + title if not title.startswith(' ') else title
    title = title + ' ' if not title.endswith(' ') else title
    tab = tabulate(*args, **kwargs)
    tit = title.center(tab.find('\n'), fill_char)
    return f"\n{tit}\n{tab}"


def clear_console():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def matched_str_head(s: str, str_list: list[str]) -> str:
    for item in str_list:
        if item.startswith(s):
            return item
    return ''


@contextmanager
def redirect_warnings():
    import warnings
    orig_show = warnings.showwarning

    def redirected_show(message, category, *args, **kwargs) -> None:
        if issubclass(category, UserWarning):
            logger.warning(message)
        else:
            orig_show(message, category, *args, **kwargs)

    warnings.showwarning = redirected_show
    try:
        yield
    finally:
        warnings.showwarning = orig_show
