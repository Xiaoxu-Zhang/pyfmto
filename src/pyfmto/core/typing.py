from typing import Literal, Union

from pyfmto.framework import AlgorithmData
from pyfmto.problem import ProblemData

TComponent = Union[AlgorithmData, ProblemData]
TComponentNames = Literal['algorithms', 'problems']
TComponentList = Union[list[AlgorithmData], list[ProblemData]]
TDiscoverResult = dict[str, dict[str, TComponentList]]
PaletteOptions = Literal[
    "deep", "deep6",
    "muted", "muted6",
    "pastel", "pastel6",
    "bright", "bright6",
    "dark", "dark6",
    "colorblind", "colorblind6",
    "Paired", "Paired6",
    "husl", "husl6",
    "Set1", "Set16",
    "Set2", "Set26",
    "Set3", "Set36"
]
