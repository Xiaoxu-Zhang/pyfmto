import copy
import numpy as np
from typing import Optional
from numpy import ndarray
from tabulate import tabulate

__all__ = ['Solution']


def _only_single_obj(func):
    def wrapped(*args, **kwargs):
        instance = args[0]
        if instance.obj != 1:
            raise AttributeError(f"Expected problem obj=1, got problem obj={instance.obj}")
        return func(*args, **kwargs)

    return wrapped


class Solution:
    def __init__(self, solution: Optional[dict] = None):
        # config
        self._dim: Optional[int] = None
        self._obj: Optional[int] = None
        self._fe_init: Optional[int] = None

        self._x: Optional[ndarray] = None
        self._y: Optional[ndarray] = None
        self._x_global: Optional[ndarray] = None
        self._y_global: Optional[ndarray] = None
        self._prev_size = 0

        if solution:
            self.__dict__.update(copy.deepcopy(solution))

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

    @property
    def size(self):
        if self._x is not None:
            return self._x.shape[0]
        return 0

    @property
    def num_updated(self):
        num = self.size - self._prev_size
        self._prev_size = self.size
        return num

    @property
    def x(self) -> Optional[np.ndarray]:
        return self._x

    @property
    def y(self) -> Optional[np.ndarray]:
        return self._y

    @property
    def x_global(self) -> Optional[np.ndarray]:
        return self._x_global

    @property
    def y_global(self) -> Optional[np.ndarray]:
        return self._y_global

    @property
    def initialized(self):
        return self._fe_init is not None

    @property
    def dim(self):
        return self._dim

    @property
    def obj(self):
        return self._obj

    @property
    def fe_init(self):
        return self._fe_init

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
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"expect x,y to have ndim=2, got x.ndim={x.ndim}, y.ndim={y.ndim} instead.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"expect x,y to have same number of rows, got (x {x.shape[0]}, y {y.shape[0]}) instead.")

        if not self.initialized:
            self._dim = x.shape[1]
            self._obj = y.shape[1]
            self._fe_init = x.shape[0]
            self._x = x
            self._y = y
        else:
            if x.shape[1] != self.dim or y.shape[1] != self.obj:
                raise ValueError(
                    f"expect x.shape=(n,{self.dim}), y.shape=(n,{self.obj}) ,got x.shape={x.shape}, y.shape={y.shape} instead")
            self._x = np.vstack((self.x, x))
            self._y = np.vstack((self.y, y))

    def clear(self):
        """
        Clear all solutions from the solution set. After calling this
        method, the solution set will be considered uninitialized.
        """
        self._dim = None
        self._obj = None
        self._x = None
        self._y = None
        self._fe_init = None
        self._prev_size = 0
