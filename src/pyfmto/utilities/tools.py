import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table, box
from tabulate import tabulate

from .loggers import logger

__all__ = [
    'add_sources',
    'clear_console',
    'colored',
    'deepmerge',
    'get_cpu_model',
    'get_os_name',
    'get_pkgs_version',
    'matched_str_head',
    'print_dict_as_table',
    'redirect_warnings',
    'tabulate_formats',
    'terminate_popen',
    'titled_tabulate'
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


def add_sources(paths: list[str]):
    import sys
    for p in paths:
        root = Path(p).resolve()
        if not root.exists():
            logger.warning(f"Path '{root}' does not exist.")
            continue
        if str(root.parent) not in sys.path:
            sys.path.append(str(root.parent))


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
    if not data:
        return
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


def get_os_name():
    os_name = sys.platform
    if os_name == 'win32':
        return 'Windows'
    elif os_name == 'darwin':
        return 'macOS'
    else:
        return 'Linux'


def get_pkgs_version(packages: list[str]):
    res: dict[str, str] = {}

    for name in packages:
        # 1. importlib.metadata (Python 3.8+)
        try:
            from importlib.metadata import version
            res[name] = version(name)
            continue
        except Exception:
            pass

        # 2. module.__version__
        from importlib import import_module
        try:
            module = import_module(name)
            res[name] = module.__version__
            continue
        except Exception:
            pass

        res[name] = 'unknown'
    return res


def get_cpu_model():
    # 1. try platform.processor()
    model = platform.processor().strip()
    if model and model not in ('', 'arm', 'x86_64'):
        return model

    # 2. macOS fallback
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            model = result.stdout.strip()
            if model:
                return model
        except Exception:
            pass

    # 3. Linux fallback
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # 4. final fallback
    return "Unknown CPU"


def deepmerge(a: dict[str, Any], b: dict[str, Any], lock: bool = True) -> dict[str, Any]:
    """
    Merge two dictionaries recursively, 'a' will be updated in place and will be returned.

    Args:
        a: Dictionary to be updated
        b: Dictionary containing values to merge
        lock: If True, only update keys that exist in 'a'; if False, also add new keys from 'b' that don't exist in 'a'

    Returns:
        Updated dictionary 'a'
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError("Both arguments must be dictionaries.")

    for key in b:
        if key not in a and lock:
            continue

        val_a = a.get(key)
        val_b = b[key]

        if key not in a and not lock:
            a[key] = val_b
        elif type(val_a) is not type(val_b):
            raise TypeError(
                f"Type mismatch at key '{key}': "
                f"a has {type(val_a).__name__}, b has {type(val_b).__name__}"
            )
        elif isinstance(val_a, dict):
            deepmerge(val_a, val_b, lock)
        else:
            a[key] = val_b

    return a

#
# def pretty_diff(a: dict, b: dict, print_it: bool = False) -> str:
#     """
#     Calculate the difference between a and b, displaying with rich in the console.
#     Mark removed content in 'a' and additions/modifications in 'b' relative to 'a'.
#     For keys newly added in 'b', mark the key and all its values in green.
#     For values modified in 'b', mark the key and new value in blue.
#     For keys removed in 'b', mark the key and all its values in red in 'a'.
#     Print in YAML format in two columns to the console with borders.
#
#     Args:
#         a: Original dictionary
#         b: Modified dictionary to compare against 'a'
#         print_it: Whether to print the comparison to console
#
#     Returns:
#         Formatted string representation of the differences
#     """
#     import yaml
#     from rich.columns import Columns
#     from rich.console import Console
#     from rich.panel import Panel
#     from rich.text import Text
#
#     def style_add(s: str) -> str:
#         return f'[green]{s}[/green]'
#
#     def style_modify(s: str) -> str:
#         return f'[blue]{s}[/blue]'
#
#     def style_remove(s: str) -> str:
#         return f'[red]{s}[/red]'
#
#     def build_marked_yaml_for_original(original, modified):
#         """Create a marked-up version of the original dict to highlight removed items."""
#         result_lines = []
#
#         # Get all keys that appear in original but not in modified (these are removed)
#         removed_keys = set(original.keys()) - set(modified.keys())
#
#         for key in sorted(original.keys()):
#             if key in removed_keys:
#                 # Key was removed in modified
#                 if isinstance(original[key], dict):
#                     result_lines.append(style_remove(f"{key}: "))
#                     sub_dict_yaml = yaml.dump(original[key], default_flow_style=False)
#                     for line in sub_dict_yaml.strip().split('\n'):
#                         if line.strip():  # Skip empty lines
#                             result_lines.append(style_remove(f'  {line}'))
#                 else:
#                     result_lines.append(style_remove(f"{key}: {original[key]}"))
#             else:
#                 # Key exists in both, show normally
#                 if isinstance(original[key], dict):
#                     result_lines.append(f'{key}: ')
#                     sub_dict_yaml = yaml.dump(original[key], default_flow_style=False)
#                     for line in sub_dict_yaml.strip().split('\n'):
#                         if line.strip():  # Skip empty lines
#                             result_lines.append(f'  {line}')
#                 else:
#                     result_lines.append(f'{key}: {original[key]}')
#
#         return '\n'.join(result_lines)
#
#     def build_marked_yaml_for_modified(original, modified):
#         """Create a marked-up version of the modified dict to highlight changes."""
#         result_lines = []
#
#         # Get all keys that appear in either dict
#         all_keys = sorted(set(original.keys()) | set(modified.keys()))
#
#         for key in all_keys:
#             if key not in original and key in modified:
#                 # Key was added
#                 if isinstance(modified[key], dict):
#                     # For nested dicts, we need to handle recursively
#                     result_lines.append(style_add(f"{key}: "))
#                     sub_dict_yaml = yaml.dump(modified[key], default_flow_style=False)
#                     for line in sub_dict_yaml.strip().split('\n'):
#                         if line.strip():  # Skip empty lines
#                             result_lines.append(style_add(f'  {line}'))
#                 else:
#                     result_lines.append(style_add(f"{key}: {modified[key]}"))
#             elif key in original and key not in modified:
#                 # Key was removed - this shouldn't appear in modified, so skip
#                 continue
#             elif original[key] != modified[key]:
#                 # Key exists in both but values differ
#                 if isinstance(modified[key], dict) and isinstance(original[key], dict):
#                     result_lines.append(style_modify(f"{key}: "))
#                     # Here we would recursively handle nested differences
#                     sub_dict_yaml = yaml.dump(modified[key], default_flow_style=False)
#                     for line in sub_dict_yaml.strip().split('\n'):
#                         if line.strip():  # Skip empty lines
#                             result_lines.append(style_modify(f'  {line}'))
#                 else:
#                     result_lines.append(style_modify(f"{key}: {original[key]} -> {modified[key]}"))
#             else:
#                 # Key exists in both and values are the same
#                 if isinstance(modified[key], dict):
#                     result_lines.append(f'{key}: ')
#                     sub_dict_yaml = yaml.dump(modified[key], default_flow_style=False)
#                     for line in sub_dict_yaml.strip().split('\n'):
#                         if line.strip():  # Skip empty lines
#                             result_lines.append(f'  {line}')
#                 else:
#                     result_lines.append(f'{key}: {modified[key]}')
#
#         return '\n'.join(result_lines)
#
#     # Create marked-up versions for both a and b
#     marked_yaml_a = build_marked_yaml_for_original(a, b)
#     marked_yaml_b = build_marked_yaml_for_modified(a, b)
#
#     # Combine for output
#     output = f"Original (A):\n{marked_yaml_a}\nModified (B):\n{marked_yaml_b}"
#
#     if print_it:
#         console = Console()
#
#         # Create panels for side-by-side display
#         panel_a = Panel(
#             Text.from_markup(marked_yaml_a),
#             title="Original (A) - Red:Removed",
#             border_style="red",
#             padding=(1, 1)
#         )
#
#         panel_b = Panel(
#             Text.from_markup(marked_yaml_b),
#             title="Modified (B) - Green:Added, Blue:Modified",
#             border_style="blue",
#             padding=(1, 1)
#         )
#
#         # Print the side-by-side comparison
#         console.print(Columns([panel_a, panel_b], equal=True))
#
#     return output
