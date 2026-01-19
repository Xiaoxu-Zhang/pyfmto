from typing import Literal, Union

from pyfmto.framework import AlgorithmData
from pyfmto.problem import ProblemData

TComponent = Union[AlgorithmData, ProblemData]
TComponentNames = Literal['algorithms', 'problems']
TComponentList = Union[list[AlgorithmData], list[ProblemData]]
TDiscoverResult = dict[str, dict[str, TComponentList]]
