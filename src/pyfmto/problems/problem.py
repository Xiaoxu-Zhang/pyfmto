import copy
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wrapt
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from numpy import ndarray
from pyDOE import lhs
from pathlib import Path
from tabulate import tabulate
from typing import List, Union, Tuple, Optional, Literal

from .solution import Solution

__all__ = ['SingleTaskProblem', 'MultiTaskProblem', 'check_and_transform']

T_Bound = Union[int, float, list, tuple, np.ndarray]


def _check_x(x, dim):
    if isinstance(x, (int, float)):
        x = np.array([[x]])
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Supported datasets types are [int, float, list, tuple, np.ndarray], got {type(x)} instead.")

    if x.ndim == 1:
        x = x.reshape(-1, dim)
    elif x.ndim > 2:
        raise ValueError(f"Expect 1<=datasets.ndim<=2, got ndim={x.ndim} instead")

    if x.shape[1] != dim:
        raise ValueError(f"expect row dim={dim}, got dim={x.shape[1]} instead")

    return x


def _transform_x(x, rot_mat, shift_mat):
    x = x - shift_mat if shift_mat is not None else x
    x = x @ rot_mat.T if rot_mat is not None else x
    return x


def check_and_transform():
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        dim = instance.dim
        x = copy.deepcopy(args[0])
        x = _check_x(x, dim)
        x = _transform_x(x, instance.rotate_mat, instance.shift_mat)
        x = np.clip(x, instance.x_lb, instance.x_ub)
        return wrapped(x, *args[1:], **kwargs)

    return wrapper


class SingleTaskProblem(ABC):
    def __init__(self, dim: int, obj: int, x_lb: T_Bound, x_ub: T_Bound, **kwargs):
        """
        Initialize a `SingleTaskProblem` instance.

        Parameters
        ----------
        dim : int
            Dimension of the decision space (number of input variables).
        obj : int
            Dimension of the objective space (number of output objectives).
        x_lb : T_Bound
            Lower bounds for the decision variables. Can be a scalar or an array-like of shape `(dim,)`.
        x_ub : T_Bound
            Upper bounds for the decision variables. Must have the same shape as [x_lb].
        **kwargs : dict, optional
            Additional keyword arguments. Supported options are:

            - init_fe : int, optional
              Initial function evaluations.
            - max_fe : int, optional
              Maximum function evaluation budget.
            - np_per_dim : int, default=1
              Number of partitions per dimension for non-IID simulation.

        Raises
        ------
        ValueError
            If invalid types or values are provided for `dim`, `obj`, `x_lb`, or `x_ub`.
        TypeError
            If unsupported argument types are passed for `x_lb` or `x_ub`.
        ValueError
            If any unrecognized keyword arguments are provided.

        Notes
        -----
        - The actual bounds are stored in `self.x_lb` and `self.x_ub`.
        - A partition can be generated if `np_per_dim > 1` to simulate non-IID scenarios.
        - The optimal solutions and Pareto Front (`self.optimum`, `self.PF`) are precomputed with default settings.
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be an integer (>0), got type({type(dim)}), value({dim}) instead")
        if not isinstance(obj, int) or obj <= 0:
            raise ValueError(f"obj must be an integer (>0), got type({type(dim)}), value({dim}) instead")

        init_fe = kwargs.get('init_fe')
        max_fe = kwargs.get('max_fe')
        np_per_dim = kwargs.get('np_per_dim', 1)

        other_args = set(kwargs.keys()) - {'init_fe', 'max_fe', 'np_per_dim'}
        if other_args:
            raise ValueError(f"got unrecognized arg(s): {other_args}")

        self._id = -1
        self._dim = dim
        self._obj = obj
        self._x_lb = np.zeros(self.dim)
        self._x_ub = np.ones(self.dim)
        self._fe_init = -1
        self._fe_max = -1
        self._np_per_dim = np_per_dim
        self._partition = np.zeros((2, self.dim))
        self._init_bounds(x_lb, x_ub)
        self._init_budget(init_fe, max_fe)
        self._solutions = Solution()

        self.rotate_mat: Optional[ndarray] = None
        self.shift_mat: Optional[ndarray] = None

    def __str__(self):
        lb, ub = self.x_lb, self.x_ub
        if np.all(self.x_lb == self.x_lb[0]) and np.all(self.x_ub == self.x_ub[0]):
            lb, ub = self.x_lb[0], self.x_ub[0]
        func_info = {
            "ID": [self.id],
            "OriFunc": [self.name],
            "DecDim": [self.dim],
            "x_lb": [lb],
            "x_ub": [ub]
        }

        tab = tabulate(func_info, headers="keys", missingval='-', tablefmt="rounded_grid")
        return tab

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"ID={self.id}, "
                f"dim={self.dim}, obj={self.obj}, "
                f"lb={self.x_lb}, ub={self.x_ub}, "
                f"init_fe={self.fe_init}, max_fe={self.fe_max})")

    def _init_bounds(self, x_lb, x_ub):
        # set bounds
        if isinstance(x_lb, (float, int)):
            lb = x_lb * np.ones(self.dim)
        else:
            lb = np.asarray(x_lb)

        if isinstance(x_ub, (float, int)):
            ub = x_ub * np.ones(self.dim)
        else:
            ub = np.asarray(x_ub)

        # check dimension
        if lb.ndim != 1 or ub.ndim != 1:
            raise ValueError(f"Lower and upper bounds must be scalars or arrays of shape({self.dim},)\n"
                             f"got shape({lb.shape}) instead")
        if lb.shape[0] != self.dim or ub.shape[0] != self.dim:
            raise ValueError(f"Lower and upper bound dimensions must match dim\n"
                             f"shape_lb={lb.shape}, shape_ub={ub.shape}, dim={self.dim}")
        # check bounds
        if not np.all(lb < ub):
            raise ValueError("Lower bound must be less than upper bound")

        self._x_lb = lb
        self._x_ub = ub

    def _init_budget(self, fe_init, fe_max):
        fe_i = fe_init if fe_init else self.dim * 5
        fe_m = fe_max if fe_max else self.dim * 11

        assert isinstance(fe_i, int), f"init_fe should be int, but type(init_fe)={type(fe_i)}"
        assert isinstance(fe_m, int), f"max_fe should be int, but type(max_fe)={type(fe_m)}"
        assert fe_i <= fe_m, f"init_fe={fe_i} > max_fe={fe_m}"

        self._fe_init = fe_i
        self._fe_max = fe_m

    def init_partition(self):
        """
        Set attribute 'partition' to simulate the non-IID settings.

        Notes
        -----
        It would do nothing if np_per_dim=1.

        - >>> Example of the bounds chosen process when dim=5 and np_per_dim=5
        - [[0.0  0.2  0.4  0.6  0.8->1.0],
        -  [0.0  0.2->0.4  0.6  0.8  1.0],
        -  [0.0  0.2->0.4  0.6  0.8  1.0],
        -  [0.0  0.2  0.4->0.6  0.8  1.0],
        -  [0.0  0.2  0.4  0.6->0.8  1.0]]
        - >>> partition will be
        - [[0.8 0.2 0.2 0.4 0.6]   <- lb
        -  [1.0 0.4 0.4 0.6 0.8]]  <- ub
        """

        if self.np_per_dim in (None, 1):
            self._np_per_dim = 1
            return
        p_mat = [np.linspace(0, 1, self.np_per_dim + 1) for _ in range(self.dim)]
        p_sampled = np.random.randint(0, self.np_per_dim, size=self.dim)
        lb = []
        ub = []
        for idx, p in zip(p_sampled, p_mat):
            lb.append(p[idx])
            ub.append(p[idx + 1])
        partition = np.array([lb, ub]) * (self.x_ub - self.x_lb) + self.x_lb
        self._partition = partition

    def visualize(self, filename: Optional[str] = None, num_points=100):
        """
        Visualize the objective function in 1D or 2D decision space.

        This method generates a plot of the objective function over the defined
        decision space. For 1D problems, a line plot is generated. For 2D problems,
        a 3D surface plot is created.

        Parameters
        ----------
        num_points : int, optional
            Number of grid points per dimension for visualization. Default is 100.
        filename : str, optional
            File path to save the generated plot. If not provided, the plot will be displayed
            interactively using `plt.show()`. Default is None.

        Raises
        ------
        ValueError
            If the problem's dimension is not 1 or 2.

        Examples
        --------
        - problem.visualize(num_points=200, path='./plots/ackley_1d.png')  # Save 1D plot
        - problem.visualize(path='./plots/rosenbrock_2d.png')  # Save 2D plot with default 100 points

        Notes
        -----
        - This method uses `matplotlib` for plotting and supports saving plots in any format
          supported by `matplotlib.pyplot.savefig`.
        """

        if self.dim not in (1, 2):
            raise ValueError(f"Visualization is only supported for 1D or 2D problems, got dim={self.dim} instead.")

        # Generate grid points for visualization
        if self.dim == 1:
            x_values = np.linspace(self.x_lb[0], self.x_ub[0], num_points)
            y_values = self.evaluate(x_values.reshape(-1, 1)).flatten()

            plt.figure(figsize=(8, 6))
            plt.plot(x_values, y_values, label='Objective Function')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'{self.name} Function Visualization (1D)')
            plt.legend()
            plt.grid(True)

        elif self.dim == 2:
            x1 = np.linspace(self.x_lb[0], self.x_ub[0], num_points)
            x2 = np.linspace(self.x_lb[1], self.x_ub[1], num_points)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.stack([X1.ravel(), X2.ravel()], axis=1)
            Y = self.evaluate(X).reshape(X1.shape)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('f(x1, x2)')
            ax.set_title(f'{self.name} Function Visualization (2D)')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

    def set_transform(self, rot_mat: Optional[np.ndarray], shift_mat: Optional[np.ndarray]):
        self.rotate_mat = rot_mat
        self.shift_mat = shift_mat

    def set_id(self, _id: int):
        self._id = _id

    def init_solutions(self):
        """
        Initialize solutions for the problem using either LHS sampling or random sampling within a partition.

        If no partition is defined (`self._partition` is None), Latin Hypercube Sampling (LHS)
        is used to generate initial solutions. Otherwise, random uniform sampling within the
        defined partition is performed.

        The generated solutions are evaluated using the objective function and stored in
        `self.solutions`.

        Returns
        -------
        None
            This method does not return any value. It initializes and stores solutions internally.

        Notes
        -----
        - The number of initial samples is determined by `self.solutions.fe_init`.
        - The sampled points are transformed from the normalized space to the original decision space.
        """
        x_init = lhs(self.dim, samples=self.fe_init)
        if self.no_partition:
            x_init = x_init * (self.x_ub - self.x_lb) + self.x_lb
        else:
            x_init = x_init * (self._partition[1] - self._partition[0]) + self._partition[0]

        y_init = np.array(self.evaluate(x_init))
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, self.obj)
        self.solutions.clear()  # Do not remove this line
        self.solutions.append(x_init, y_init)

    def random_uniform_x(self, size, within_partition=True):
        """
        Sample points uniformly within the hypercube defined by :math:`(x_{lb}, x_{ub})^{dim}`.

        Parameters
        ----------
        size : int
            Number of samples, int
        within_partition : bool
            Within partition or not

        Returns
        -------
        samples : np.ndarray
            Samples
        """

        if not self.no_partition and within_partition:
            return np.random.uniform(self._partition[0], self._partition[1], size=(size, self.dim))
        else:
            return np.random.uniform(low=self.x_lb, high=self.x_ub, size=(size, self.dim))

    def normalize_x(self, points):
        """
        Normalize a set of points to the :math:`(0, 1)^D` space, where
        :math:`D` is the dimension of the problem's decision space.

        Parameters
        ----------
        points : np.ndarray or list or tuple
            A set of points.
        Returns
        -------
        points : np.ndarray
            Normalized points
        """
        self._check_inputs(points)
        if not np.all(self.x_lb <= points) or not np.all(points <= self.x_ub):
            points = np.clip(points, self.x_lb, self.x_ub)
            # ologger.warning(f"x_lb is {self.x_lb}, x_ub is {self.x_ub}, but not all points are in the range, "
            #                f"np.clip() is applied to ensure all values are within the range.")
        return (points - self.x_lb) / (self.x_ub - self.x_lb)

    def denormalize_x(self, points):
        """
        Denormalize a set of points to function original decision space.
        You may want to set normalize=True if you wish the function's
        decision variables :math:`x \\in (0,1)^d`
        Parameters
        ----------
        points : np.ndarray or list or tuple
            A set of points. :math:`x \\in (0,1)^d`
        Returns
        -------
        points : np.ndarray
            Denormalized points
        """
        points = self._check_inputs(points)
        if not np.all(0 <= points) or not np.all(points <= 1):
            points = np.clip(points, 0, 1)
            # ologger.warning("Points to be denormalized contain values outside the range [0, 1]. "
            #                "np.clip() is applied to ensure all values are within the range.")
        return points * (self.x_ub - self.x_lb) + self.x_lb

    @abstractmethod
    def evaluate(self, x, *args, **kwargs):
        """
        OjbFunc:= :math:`\\mathbf{x} \\to f \\to \\mathbf{y}, \\mathbf{x} \\in \\mathbb{R}^{D1},
        \\mathbf{y}\\in \\mathbb{R}^{D2}`, where :math:`D1` is the dimension of decision space
        and :math:`D2` is that of objective space.

        Parameters
        ----------
        x: np.ndarray
            Inputs of the problem.
        args
        kwargs

        Returns
        -------
        np.ndarray or float
            Outputs corresponding to the problem inputs.
        """

    def _check_inputs(self, points):
        return _check_x(points, self.dim)

    @property
    def no_partition(self):
        return np.all(self._partition == 0)

    @property
    def id(self) -> int:
        return self._id

    @property
    def x_lb(self) -> ndarray:
        return self._x_lb

    @property
    def x_ub(self) -> ndarray:
        return self._x_ub

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def obj(self):
        return self._obj

    @property
    def fe_init(self):
        return self._fe_init

    @property
    def fe_max(self):
        return self._fe_max

    @property
    def fe_available(self):
        return self.fe_max - self.solutions.size

    @property
    def np_per_dim(self) -> Optional[int]:
        return self._np_per_dim

    @property
    def solutions(self) -> Solution:
        return self._solutions


T_Tasks = Union[List[SingleTaskProblem], Tuple[SingleTaskProblem]]


class MultiTaskProblem(ABC):
    def __init__(self, is_realworld, *args, **kwargs):
        self.is_realworld = is_realworld
        self.seed = kwargs.pop('seed', 123)
        self._problem = self.__init_tasks(*args, **kwargs)

    def __init_tasks(self, *args, **kwargs):
        problem = self._init_tasks(*args, **kwargs)
        if not isinstance(problem, (list, tuple)):
            raise TypeError(f"init_tasks() must return a list or tuple, got {type(problem)} instead.")
        self._check_problem(problem)
        return problem

    @staticmethod
    def _check_problem(problem: T_Tasks):
        if problem[0].id == -1:
            for i in range(len(problem)):
                problem[i].set_id(i + 1)

    @abstractmethod
    def _init_tasks(self, *args, **kwargs) -> T_Tasks: ...

    def __iter__(self):
        return iter(self._problem)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._problem[index]
        elif isinstance(index, int):
            if index < 0:
                raise IndexError(f"Index {index} < 0 is invalid.")
            if index >= len(self._problem):
                raise IndexError(f"Index {index} > total tasks={len(self._problem)} in problem {self.problem_name}")
            return self._problem[index]
        else:
            raise TypeError(f"Index must be an integer or a slice, but {type(index)} given.")

    def init_solutions(self, random_ctrl: Literal['no', 'weak', 'strong']):
        if random_ctrl == 'no':
            self._init_partition()
            self._init_solutions()
        elif random_ctrl == 'weak':
            self._set_seed()
            self._init_partition()
            self._unset_seed()
            self._init_solutions()
        elif random_ctrl == 'strong':
            self._set_seed()
            self._init_partition()
            self._init_solutions()
            self._unset_seed()
        else:
            raise ValueError('Invalid random_ctrl option.')

    def _init_partition(self):
        for p in self._problem:
            p.init_partition()

    def _init_solutions(self):
        for p in self._problem:
            p.init_solutions()

    def _set_seed(self):
        self._rdm_state = np.random.get_state()
        np.random.seed(self.seed)

    def _unset_seed(self):
        np.random.set_state(self._rdm_state)

    def show_distribution(self, filename=None):
        init_x = {f"T{p.id}({p.name})": p.normalize_x(p.solutions.x) for p in self}
        np_per_dim = self[0].np_per_dim
        tick_interval = 1.0 / np_per_dim
        ticks = np.arange(0, 1 + tick_interval, tick_interval)
        data = []
        for label, points in init_x.items():
            for point in points:
                if point.shape[0] >= 2:
                    data.append({
                        'x': point[0],
                        'y': point[1],
                        'problem': label
                    })
        sns.set(style="whitegrid")
        plt.figure(figsize=(11, 9))
        sns.scatterplot(data=pd.DataFrame(data), x='x', y='y', hue='problem', style='problem', s=80)
        plt.title(f'Initial Solutions of Each Task (np_per_dim={np_per_dim})')
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.legend(title=f'Tasks', bbox_to_anchor=(1., 1), loc='upper left')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    @property
    def problem_name(self) -> str:
        return self.__class__.__name__

    @property
    def task_num(self):
        return len(self._problem)

    def __len__(self):
        return len(self._problem) if self._problem is not None else 0

    def __repr__(self):
        t = "realworld" if self.is_realworld else "synthetic"
        return f"{self.problem_name}({len(self._problem)} {t} tasks)"

    def __str__(self):
        info_list = {
            "ProbName": [self.problem_name],
            "ProbType": ['Realworld' if self.is_realworld else 'Synthetic'],
            "TaskNum": [len(self._problem)],
            "DecDim": [self._problem[0].dim],
            "ObjDim": [self._problem[0].obj]
        }
        return tabulate(info_list, headers="keys", tablefmt="rounded_grid")
