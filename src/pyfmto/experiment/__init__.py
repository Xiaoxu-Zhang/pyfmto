from ..utilities.io import dumps_yaml, parse_yaml
from ..utilities.tools import matched_str_head
from .config import ConfigLoader, LauncherConfig, ReporterConfig
from .launcher import Launcher
from .reporter import Reporter
from .utils import (
    RunSolutions,
)

__all__ = [
    'ConfigLoader',
    'Launcher',
    'LauncherConfig',
    'Reporter',
    'ReporterConfig',
    'RunSolutions',
    'list_report_formats',
    'show_default_conf',
]


def list_report_formats(print_it=False):
    res = [r[3:] for r in dir(Reporter) if 'to_' in r]
    if print_it:
        print('\n'.join(res))
    return res


def show_default_conf(name: str):
    conf = parse_yaml(DEFAULT_CONF)
    full_name = matched_str_head(name, list(conf.keys()))
    if full_name:
        print(dumps_yaml(conf[full_name]))
    else:
        print(f"No matched format name {name} in report formats {list(conf.keys())}")


DEFAULT_CONF = """
curve:
    suffix: .png  # File extension for the generated curve image. options: '.png', '.jpg', '.eps', '.svg', '.pdf'
    quality: 3  # int in [1, 9]. Image quality parameter, affecting the quality of scalar images.
    showing_size: -1  # Use 'last showing_size data points' to plot curve. If -1, use all data points.
    merge: True  # Merge all plotted images to a single figure or not. Only support for '.png' and '.jpg' files.
    clear: True  # If True, clear plots after merge. Only applicable when merge is True.
    on_log_scale: False  # The plot is generated on a logarithmic scale.
    alpha: 0.2  # Transparency of the Standard Error region, 0 to 1 (completely to opaque).
violin:
    suffix: .png  # File extension for the generated violin image. Options: '.png', '.jpg', '.eps','.svg', '.pdf'.
    quality: 3   # int in [1, 9]. Image quality parameter, affecting the quality of scalar images.
    merge: True  # Merge all plotted images to a single figure or not. Only support for '.png' and '.jpg' files.
    clear: True  # If True, clear plots after merge. Only applicable when merge is True.
console:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
excel:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
latex:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
"""
