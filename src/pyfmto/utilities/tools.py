import os
import platform
import time
import wrapt
from pyfmto.utilities import logger
from tabulate import tabulate
from typing import Literal, Optional

__all__ = [
    'colored',
    'timer',
    'clear_console',
    'show_in_table',
    'update_kwargs',
    'titled_tabulate',
    'tabulate_formats',
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


def timer(name: Optional[str] = None, where: Literal['log', 'console', 'both'] = 'log'):
    """
    A decorator that records the execution time of a function and outputs it to log, console, or both.

    This decorator wraps a function and measures its execution time. The result can be output to a logger,
    the console (with optional color formatting), or both. An optional custom name can be provided for
    the function in the output message; otherwise, the function's own name is used.

    Parameters
    ----------
    name :
        Custom name to display for the decorated function in the output message.
        If not provided, the function's name will be used.

    where :
        Specifies where to output the runtime message:

        - 'log': Output only to the logger (`ologger.info`).
        - 'console': Output only to the terminal with green-colored formatting.
        - 'both': Output to both the logger and the console. Default is `'log'`.

    Returns
    -------
    function
        A wrapped version of the original function that includes timing behavior.

    Notes
    -----
    - Elapsed time is measured in seconds with three decimal places precision.
    - This decorator uses the `wrapt` library to ensure compatibility with various function types
      and to preserve metadata (e.g., docstrings and function names).

    Examples
    --------
    Example 1: Basic usage with default options

    >>> @record_runtime()
    ... def example_func():
    ...     import time
    ...     time.sleep(0.5)
    ...
    >>> example_func()
    INFO - =============== example_func cost 0.500 s ===============

    Example 2: Custom name and console output

    >>> @record_runtime(name="CustomTask", where="console")
    ... def task():
    ...     pass
    ...
    >>> task()
    =============== CustomTask cost 0.000 s ===============

    Example 3: Output to both log and console

    >>> @record_runtime(where='both')
    ... def demo():
    ...     time.sleep(0.1)
    ...
    >>> demo()
    INFO - =============== demo cost 0.100 s ===============
    =============== demo cost 0.100 s ===============
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        start_time = time.time()
        result = wrapped(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        if name is not None:
            _name = name
        else:
            _name = wrapped.__name__
        _log = f"{'=' * 15} {_name} cost {runtime:.3f} s {'=' * 15}"
        _console = colored(_log, 'green')
        if where == 'log':
            logger.info(_log)
        elif where == 'console':
            print(_console)
        else:
            logger.info(_log)
            print(_console)
        return result

    return wrapper


def titled_tabulate(title: str, fill_char: str, *args, **kwargs):
    title = ' ' + title if not title.startswith(' ') else title
    title = title + ' ' if not title.endswith(' ') else title
    tab = tabulate(*args, **kwargs)
    tit = title.center(tab.find('\n'), fill_char)
    return f"\n{tit}\n{tab}"


def show_in_table(**kwargs):
    keys, colored_values, original_values = zip(*map(_mapper, kwargs.items()))
    colored_data = dict(zip(keys, colored_values))
    original_data = dict(zip(keys, original_values))
    alignment = ['center'] * len(kwargs)
    colored_tab = tabulate(colored_data, headers='keys', tablefmt='rounded_grid', colalign=alignment)
    original_tab = tabulate(original_data, headers='keys', tablefmt='rounded_grid', colalign=alignment)
    return colored_tab, original_tab


def _mapper(item):
    k, v = item
    if v is True:
        return k, [colored('yes', 'green')], [str(v)]
    elif v is False:
        return k, [colored('no', 'red')], [str(v)]
    elif isinstance(v, int):
        return k, [colored(str(v), 'magenta')], [str(v)]
    elif isinstance(v, float):
        val_str = f"{v:.3f}"
        return k, [colored(val_str, 'green')], [val_str]
    else:
        return k, [str(v)], [str(v)]


def update_kwargs(name, defaults: dict, updates: dict):
    """
    Update ``defaults``  with values from ``updates``.

    This function takes a set of default parameters and a set of updated parameters,
    merges them (with updates taking precedence), and logs a formatted table showing
    the differences. It's useful for configuration management where you want to track
    what values are being used and how they differ from defaults.

    Parameters
    ----------
    name : str
        The name/title for the parameter update operation, used in logging.
    defaults : dict
        Dictionary containing default parameter values.
    updates : dict
        Dictionary containing updated parameter values that override defaults.

    Returns
    -------
    dict
        A new dictionary containing the merged parameters (``defaults`` updated with values from ``updates``).

    Examples
    --------
    >>> defaults = {'a': 0.1, 'b': 10}
    >>> updates = {'a': 0.05, 'c': 5}
    >>> result = update_kwargs("Training Config", defaults, updates)
    >>> print(result)
    {'a': 0.05, 'b': 10}
    """
    if not defaults and not updates:
        return {}
    _log_diff(name, defaults, updates)
    res = defaults.copy()
    for key, value in updates.items():
        if key in defaults:
            res[key] = value
    return res


def _log_diff(name, defaults: dict, updates: dict):
    table_data = []
    for key in set(defaults.keys()).union(updates.keys()):
        default_val = str(defaults[key]) if key in defaults else '-'
        updates_val = str(updates[key]) if key in updates else '-'
        if key in defaults:
            using_val = updates.get(key, defaults[key])
        else:
            using_val = '-'
        table_data.append([key, default_val, updates_val, str(using_val)])
    table = titled_tabulate(
        name,
        '=',
        table_data,
        headers=["Parameter", "Default", "Updates", "Using"],
        tablefmt="rounded_grid",
        colalign=("left", "center", "center", "center")
    )
    logger.info(table)


def clear_console():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')
