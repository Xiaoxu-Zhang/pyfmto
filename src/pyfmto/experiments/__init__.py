from .reporter import Reports
from .launcher import Launcher
from .utils import (
    RunSolutions,
)

__all__ = [
    'Reports',
    'Launcher',
    'RunSolutions',
    'list_reports',
    'DEFAULT_CONF',
]


def list_reports(print_it=False):
    res = [r[3:] for r in dir(Reports) if 'to_' in r]
    if print_it:
        print('\n'.join(res))
    return res


DEFAULT_CONF = """
curve:
    suffix: .png  # File extension for the generated curve image. options: '.png', '.jpg', '.eps', '.svg', '.pdf'
    quality: 3  # int in [1, 9]. Image quality parameter, affecting the quality of scalar images.
    showing_size: -1  # Use 'last showing_size data points' to plot curve. If -1, use all data points.
    merge: True  # Merge all plotted images to a single figure or not. Note that only '.png' and '.jpg' files can be merged.
    clear: True  # If True, clear plots after merge. Only applicable when merge is True.
    on_log_scale: False  # The plot is generated on a logarithmic scale.
    alpha: 0.2  # Transparency of the Standard Error region, ranging from 0 (completely transparent) to 1 (completely opaque).
violin:
    suffix: .png  # File extension for the generated violin image. Options: '.png', '.jpg', '.eps','.svg', '.pdf'.
    quality: 3   # int in [1, 9]. Image quality parameter, affecting the quality of scalar images.
    merge: bool  # Merge all plotted images to a single figure or not. Note that only '.png' and '.jpg' files can be merged.
    clear: True  # If True, clear plots after merge. Only applicable when merge is True.
console:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
excel:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
latex:
    pvalue: 0.5 # T-test threshold for determining statistical significance.
"""