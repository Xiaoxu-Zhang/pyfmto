import numpy as np
from typing import Union
from numpy import ndarray
from tabulate import tabulate

from pyfmto.utilities.schemas import STPConfig

__all__ = ['Solution']


def _only_single_obj(func):
    def wrapped(*args, **kwargs):
        instance = args[0]
        if instance.obj != 1:
            raise AttributeError(f"Expected problem obj=1, got problem obj={instance.obj}")
        return func(*args, **kwargs)

    return wrapped


class Solution:
    _dim: int
    _obj: int
    _fe_init: int
    _fe_max: int
    _np_per_dim: int
    _prev_size: int
    _lb: ndarray
    _ub: ndarray
    _x: ndarray
    _y: ndarray
    _x_global: ndarray
    _y_global: ndarray

    def __init__(self, data: Union[STPConfig, dict]):
        self._prev_size = 0
        if isinstance(data, dict):
            self._init_from_dict(data)
        elif isinstance(data, STPConfig):
            self._init_from_config(data)
        else:
            raise TypeError(f"Solution() must be initialized with a dict or STPConfig, got {type(data)} instead.")

    def __repr__(self):
        return f"SolutionSet(dim={self.dim}, obj={self.obj}, solution_size={self.size})"

    def __str__(self):
        solution_info = {
            'dim': [self.dim],
            'obj': [self.obj],
            'init': [self.fe_init],
            'total': [self.size]
        }
        return tabulate(solution_info, headers='keys', tablefmt='rounded_grid')

    def to_dict(self):
        return self.__dict__

    def _init_from_dict(self, data: dict):
        required_keys = set(self.required_keys)
        actual_keys = set(data.keys())
        missing_keys = required_keys - actual_keys
        extra_keys = actual_keys - required_keys
        if missing_keys or extra_keys:
            error_msg = "Invalid data:\n"
            error_msg += f"  Expected keys: {sorted(required_keys)}"
            error_msg += f"  Missing keys: {sorted(missing_keys)}\n"
            error_msg += f"  Extra keys: {sorted(extra_keys)}\n"
            raise ValueError(error_msg)
        self.__dict__.update(data.copy())

    def _init_from_config(self, config: STPConfig):
        self._dim = config.dim
        self._obj = config.obj
        self._fe_init = config.fe_init
        self._fe_max = config.fe_max
        self._np_per_dim = config.np_per_dim
        self._lb = config.lb  # type: ignore
        self._ub = config.ub  # type: ignore
        self._x = np.array([])
        self._y = np.array([])
        self._x_global = np.array([])
        self._y_global = np.array([])

    @property
    def required_keys(self):
        return sorted(self.__class__.__annotations__.keys())

    @property
    def size(self):
        return self._x.shape[0]

    @property
    def num_updated(self):
        num = self.size - self._prev_size
        self._prev_size = self.size
        return num

    @property
    def x(self) -> ndarray:
        return self._x

    @property
    def y(self) -> ndarray:
        return self._y

    @property
    def x_global(self) -> ndarray:
        return self._x_global

    @property
    def y_global(self) -> ndarray:
        return self._y_global

    @property
    def initialized(self):
        return self.size >= self.fe_init

    @property
    def dim(self):
        return self._dim

    @property
    def obj(self):
        return self._obj

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def fe_init(self):
        return self._fe_init

    @property
    def fe_max(self):
        return self._fe_max

    @property
    def np_per_dim(self) -> int:
        return self._np_per_dim

    @property
    @_only_single_obj
    def y_max(self):
        return np.max(self.y)

    @property
    @_only_single_obj
    def y_min(self):
        return np.min(self.y)

    @property
    @_only_single_obj
    def x_of_y_max(self):
        idx = np.argmax(self.y.flatten())
        return self.x[idx]

    @property
    @_only_single_obj
    def x_of_y_min(self):
        idx = np.argmin(self.y.flatten())
        return self.x[idx]

    @property
    @_only_single_obj
    def y_homo_decrease(self):
        y_arr = self.y.flatten()
        res = [y_arr[0]]
        for y in y_arr[1:]:
            if y <= res[-1]:
                res.append(y)
            else:
                res.append(res[-1])
        return np.array(res)

    @property
    @_only_single_obj
    def y_homo_increase(self):
        y_arr = self.y.flatten()
        res = [y_arr[0]]
        for y in y_arr[1:]:
            if y >= res[-1]:
                res.append(y)
            else:
                res.append(res[-1])
        return np.array(res)

    def append(self, x: ndarray, y: ndarray):
        """
        Append one or more solutions to the solution set.
        """
        x = np.array(x)
        y = np.array(y)
        if x.ndim != 2:
            x = x.reshape(-1, self.dim)
        if y.ndim != 2:
            y = y.reshape(-1, self.obj)
        if x.shape[0] != y.shape[0]:
            raise ValueError("expect x,y to have same number of rows, "
                             f"got (x {x.shape[0]}, y {y.shape[0]}) instead.")

        if self.size == 0:
            self._x = x
            self._y = y
        else:
            self._x = np.vstack((self.x, x))
            self._y = np.vstack((self.y, y))
