from typing import Union, Literal, TypeVar

from pyfmto.framework import AlgorithmData
from pyfmto.problem import ProblemData

T = TypeVar('T')

ListOrTuple = Union[list[T], tuple[T, ...]]
TComponent = Union[AlgorithmData, ProblemData]
TComponentNames = Literal['algorithms', 'problems']
TComponentList = Union[ListOrTuple[AlgorithmData], ListOrTuple[ProblemData]]
TDiscoverResult = dict[str, dict[str, TComponentList]]
