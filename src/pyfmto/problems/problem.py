import copy
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from pyDOE import lhs
from pydantic import BaseModel, field_validator, model_validator
from scipy.stats import kendalltau, spearmanr, pearsonr

from pyfmto.utilities import StrColors, Cmaps, colored
from tabulate import tabulate
from typing import List, Union, Tuple, Optional, Literal
from .solution import Solution

__all__ = [
    'SingleTaskProblem', 'MultiTaskProblem', 'Transformer',
    'STPConfig', 'TransformerConfig']

from ..utilities.stroptions import SeabornStyles

T_Bound = Union[int, float, list, tuple, np.ndarray]


class STPConfig(BaseModel):
    """
    Basically, this model makes the following validation and convertion:

    - ``dim``: a positive integer
    - ``obj``: a positive integer
    - ``lb``: finally a ndarray with shape (dim,)
    - ``ub``: finally a ndarray with shape (dim,)
    - ``fe_init``: default to ``5*dim`` or a positive integer
    - ``fe_max``: default to ``11*dim`` or a positive integer
    - ``np_per_dim``: default to 1 or a positive integer

    Notes
    -----
    Additionally, this model validate that:
        1. bounds satisfied ``lb<ub`` on each dimension.
        2. ``fe_init<=fe_max``
    """
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}
    dim: int
    obj: int
    lb: T_Bound
    ub: T_Bound
    fe_init: Optional[int] = None
    fe_max: Optional[int] = None
    np_per_dim: Optional[int] = None

    @field_validator('dim', 'obj')
    def validate_positive_integer(cls, v):
        if v <= 0:
            raise ValueError('dim and obj must be positive integers')
        return v

    @field_validator('fe_init', 'fe_max', 'np_per_dim')
    def validate_positive_or_none(cls, v):
        if v is not None and v < 1:
            raise ValueError(f"Invalid value: {v} (fe_init, fe_max, and np_per_dim must be positive or None)")
        return v

    @model_validator(mode='after')
    def validate_model(self):
        # Convert lb and ub to dim-dimensional arrays if they are int or float
        if isinstance(self.lb, (int, float)):
            self.lb = np.ones(self.dim) * self.lb
        else:
            self.lb = np.asarray(self.lb)

        if isinstance(self.ub, (int, float)):
            self.ub = np.ones(self.dim) * self.ub
        else:
            self.ub = np.asarray(self.ub)

        # Check dimensionality
        if self.lb.shape != (self.dim,):
            raise ValueError(f"lb must be a scalar or array of shape ({self.dim},)")
        if self.ub.shape != (self.dim,):
            raise ValueError(f"ub must be a scalar or array of shape ({self.dim},)")

        # Check bounds
        if not np.all(self.lb < self.ub):
            raise ValueError("All elements of lb must be less than corresponding elements of ub")

        # Set default values for fe_init and fe_max
        if self.fe_init is None:
            self.fe_init = 5 * self.dim
        if self.fe_max is None:
            self.fe_max = 11 * self.dim

        if self.np_per_dim is None:
            self.np_per_dim = 1

        if self.fe_init > self.fe_max:
            raise ValueError("fe_init must be less than or equal to fe_max")

        return self


class TransformerConfig(BaseModel):
    """
    Basically, this model makes the following validation and conversion:

    - ``dim``: a positive integer (has been validated by STPConfig)
    - ``shift``: finally a ndarray with shape (dim,)
    - ``rotation``: finally a ndarray with shape (dim, dim)
    - ``rotation_inv``: set to inverse of rotation matrix
    """
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}
    dim: int
    shift: Optional[Union[int, float, np.ndarray]] = None
    rotation: Optional[np.ndarray] = None
    rotation_inv: Optional[np.ndarray] = None

    @model_validator(mode='after')
    def check_rotation_and_shift(self):
        if self.rotation is None:
            self.rotation = np.eye(self.dim)
            self.rotation_inv = np.eye(self.dim)
        if self.shift is None:
            self.shift = np.zeros(self.dim)
        elif isinstance(self.shift, (int, float)):
            self.shift = np.ones(self.dim) * self.shift

        if self.rotation.shape != (self.dim, self.dim):
            raise ValueError(f"{self.dim} dimensional task's rotation shape must be ({self.dim}, {self.dim})")
        if self.shift.shape != (self.dim,):
            raise ValueError(f"{self.dim} dimensional task's shift shape must be ({self.dim},)")
        self.rotation = self.rotation.T
        self.rotation_inv = np.linalg.inv(self.rotation.T)

        return self


def check_x(x, dim):
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Only support ndarray as input, got {type(x).__name__}={x} instead.")

    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"expect dim={dim}, got dim={x.shape[0]} instead")
    elif x.ndim == 2:
        if x.shape[1] != dim:
            raise ValueError(f"expect dim={dim}, got dim={x.shape[1]} instead")
    else:
        raise ValueError(f"Expect 1<=datasets.ndim<=2, got ndim={x.ndim} instead")

    return x


class Transformer:
    def __init__(self, dim: int):
        self._transform = TransformerConfig(dim=dim)

    @property
    def dim(self):
        return self._transform.dim

    @property
    def shift(self):
        return self._transform.shift

    @property
    def rotation(self):
        return self._transform.rotation

    @property
    def rotation_inv(self):
        return self._transform.rotation_inv

    def set_transform(self, rotation: Optional[np.ndarray]=None, shift: Union[int, float, np.ndarray, None]=None):
        self._transform = TransformerConfig(
            dim=self.dim,
            rotation=rotation,
            shift=shift
        )

    def transform_x(self, x):
        return (x - self.shift) @ self.rotation

    def inverse_transform_x(self, x):
        return (x @ self.rotation_inv) + self.shift


class SingleTaskProblem(ABC):
    def __init__(self, dim: int, obj: int, lb: T_Bound, ub: T_Bound, **kwargs):
        """
        Initialize a `SingleTaskProblem` instance.

        Parameters
        ----------
        dim : int
            Dimension of the decision space (number of input variables).
        obj : int
            Dimension of the objective space (number of output objectives).
        lb : T_Bound
            Lower bounds for the decision variables. Can be a scalar or an array-like of shape (dim,).
        ub : T_Bound
            Upper bounds for the decision variables. Must have the same shape as [x_lb].
        **kwargs : dict, optional
            Additional keyword arguments. Supported options are:

            - fe_init : int, optional
              Initial function evaluations.
            - fe_max : int, optional
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
        self._config = STPConfig(dim=dim, obj=obj, lb=lb, ub=ub, **kwargs)
        self._id = -1
        self._x_global = np.zeros(self.dim)
        self._partition = np.zeros((2, self.dim))
        self._solutions = Solution()
        self._transformer = Transformer(self.dim)
        self.auto_update_solutions = False

    def set_x_global(self, x_global: Optional[ndarray]):
        """
        Setting the global optimum solution x.

        Notes
        -----
        Default to zero vector if it doesn't set,
        pass a None if it is unknown.
        """
        self._x_global = x_global

    def __str__(self):
        lb, ub = self.lb, self.ub
        if np.all(self.lb == self.lb[0]) and np.all(self.ub == self.ub[0]):
            lb, ub = self.lb[0], self.ub[0]
        func_info = {
            "ID": [self.id],
            "OriFunc": [self.name],
            "DecDim": [self.dim],
            "lb": [lb],
            "ub": [ub]
        }

        tab = tabulate(func_info, headers="keys", missingval='-', tablefmt="rounded_grid")
        return tab

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"ID={self.id}, "
                f"dim={self.dim}, obj={self.obj}, "
                f"lb={self.lb}, ub={self.ub}, "
                f"init_fe={self.fe_init}, max_fe={self.fe_max})")

    def init_partition(self):
        """
        Set attribute 'partition' according to 'np_per_dim' to simulate the non-IID settings.

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
        p_mat = [np.linspace(0, 1, self.np_per_dim + 1) for _ in range(self.dim)]
        p_sampled = np.random.randint(0, self.np_per_dim, size=self.dim)
        lb = []
        ub = []
        for idx, p in zip(p_sampled, p_mat):
            lb.append(p[idx])
            ub.append(p[idx + 1])
        partition = np.array([lb, ub]) * (self.ub - self.lb) + self.lb
        self._partition = partition

    def gen_plot_data(
            self,
            dims=(0, 1),
            n_points=100,
            fixed=None,
            scale_mode: Literal['xy', 'y']='y',
        ) -> tuple:
        """

        Parameters
        ----------
        dims:
            Selected dimensions to calculate y.
        n_points:
            Number of points in one dimension, total use n_points**2 points.
        fixed:
            Fixed value of no-selected dimensions
        scale_mode:
            if pass 'xy' Normalize both the decision space and objective space to (0, 1)
            if pass 'y' Scale objective value to match the decision space, which will keep
            the shift information
        Returns
        -------

        """
        dim1, dim2 = sorted(dims)
        if not dim1 in range(self.dim) or not dim2 in range(self.dim):
            raise ValueError(f"The selected_dims's values should in [0, {self.dim - 1}].")
        lb1, ub1 = self.lb[dim1], self.ub[dim1]
        lb2, ub2 = self.lb[dim2], self.ub[dim2]
        d1 = np.linspace(lb1, ub1, n_points)
        d2 = np.linspace(lb2, ub2, n_points)
        D1, D2 = np.meshgrid(d1, d2)
        _fixed = (self.lb + self.ub) / 2 if fixed is None else fixed
        points = np.ones(shape=(n_points, n_points, self.dim)) * _fixed
        points[:, :, dim1] = D1
        points[:, :, dim2] = D2
        Z = np.apply_along_axis(self.evaluate, axis=-1, arr=points)
        Z = Z.squeeze()
        if scale_mode == 'xy':
            dn = np.linspace(0, 1, n_points)
            D1, D2 = np.meshgrid(dn, dn)
            Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        elif scale_mode == 'y':
            x_range = self.ub[0] - self.lb[0]
            Z = x_range * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        else:
            print(f"Unknown scale_mode '{scale_mode}', which should "
                  f"be '{colored('xy', 'green')}' "
                  f"or '{colored('y', 'green')}'")
        return D1, D2, Z

    def plot_2d(
            self,
            filename: Optional[str]=None,
            dims: Union[list[int, int], tuple[int, int]]=(0, 1),
            n_points: int=100,
            figsize: tuple[float, float, float]=(5., 4., 1.),
            cmap: Union[str, Cmaps]=Cmaps.viridis,
            levels: int=30,
            labels: tuple[Optional[str], Optional[str]]=(None, None),
            title: str=None,
            alpha: float=0.7,
            fixed: Union[float, np.ndarray, None]=None):
        """
        Plot the function's 2D contour.

        Parameters
        ----------
        filename:
            Filename, show image if pass None
        dims : list, tuple
            The selected dimensions indices to be drawn.
        n_points : int
            Using total n_points**2 points to draw the contour
        figsize : tuple
            figsize (width, height, scale)
        cmap : str
            The cmap of matplotlib.pyplot.contourf function. The `pyfmto.utilities.Cmaps` class
            can help you try different options easier.
        levels : int
            The levels parameter of contourf function
        labels : tuple
            The figure x, y labels.
        title :
            The figure title
        alpha : float
            The alpha parameter of contourf function
        fixed : float, np.ndarray
            Fixed values for all the unused dimensions, if pass a ndarray, its shape should be (dim,)
        """
        if self.dim < 2:
            raise ValueError(f"Only supported for 'dim>1' problems, got dim={self.dim} instead.")
        if not isinstance(dims, (list, tuple)):
            raise ValueError('dims should be [int, int] or (int, int)')
        w, h, s = figsize
        figsize = (w * s, h * s)
        plt.figure(figsize=figsize)
        D1, D2, Z = self.gen_plot_data(dims, n_points, fixed)
        cont = plt.contourf(D1, D2, Z, levels=levels, cmap=str(cmap), alpha=alpha)
        cbar = plt.colorbar(cont)
        cbar.set_label('Function Value')
        cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.2e}' if abs(x) > 1e4 else f'{x:.2f}')
        cbar.update_ticks()
        xl, yl = labels
        xl = f'X{min(dims)}' if xl is None else xl
        yl = f'X{max(dims)}' if yl is None else yl
        _title = f'T{self.id} {self.name}' if title is None else title
        plt.xlabel(xl)
        plt.ylabel(yl, rotation=0)
        plt.title(_title)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

    def plot_3d(
            self,
            filename: Optional[str] = None,
            dims: tuple[int, int]=(0, 1),
            n_points: int = 100,
            figsize: tuple[float, float, float] = (5., 4., 1.),
            cmap: Union[str, Cmaps]=Cmaps.viridis,
            levels: int = 30,
            labels: tuple[Optional[str], Optional[str], Optional[str]]=(None, None, None),
            title: str=None,
            alpha=0.7,
            fixed=None,
    ):
        """
        Plot the function's 3D surface.

        Parameters
        ----------
        filename:
            Filename, show image if pass None.
        dims : list, tuple
            The selected dimensions indices to be drawn.
        n_points : int
            Using total n_points**2 points to draw the contour.
        figsize : tuple
            figsize (width, height, scale).
        cmap : Union[str, Cmaps]
            The cmap of matplotlib.pyplot.contourf function, use Cmaps for easy selection.
        levels : int
            The levels of the contourf.
        labels : tuple
            The figure x, y, z labels.
        title :
            The figure title.
        alpha : float
            The alpha parameter of contourf function.
        fixed : float, np.ndarray
            Fixed values for all the unused dimensions, if pass a ndarray, its shape should be (dim,).
        """
        if self.dim < 2:
            raise ValueError(f"Only supported for 'dim>1' problems, got dim={self.dim} instead.")
        w, h, s = figsize
        figsize = (w * s, h * s)
        D1, D2, Z = self.gen_plot_data(dims, n_points, fixed)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(D1, D2, Z, cmap=str(cmap), edgecolor='none', alpha=alpha)
        ax.contour(D1, D2, Z, zdir='z', offset=np.min(Z), cmap=str(cmap), levels=levels)
        cbar = fig.colorbar(surf, ax=ax)
        cbar.set_label('Function Value')
        cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.2e}' if abs(x) > 1e4 else f'{x:.2f}')
        cbar.update_ticks()

        dim1, dim2 = sorted(dims)
        xl, yl, zl = labels
        xl = f'X{dim1}' if xl is None else xl
        yl = f'X{dim2}' if yl is None else yl
        zl = 'Y' if zl is None else zl
        _title = f"T{self.id} {self.name}" if title is None else title
        ax.view_init(elev=20, azim=-120)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_zlabel(zl)
        ax.set_title(_title)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

    def plot_3d_interactive(
            self,
            dims: tuple[int, int] = (0, 1),
            n_points: int = 100,
            font_size: float = 10,
            cmap: Union[str, Cmaps] = Cmaps.viridis,
            color: Union[str, StrColors, list[float, float, float], tuple[float, float, float], None]=None,
            show_grid: bool = True,
            scale_mode: Literal['xy', 'y'] = 'y',
            fixed=None,
            plotter=None
    ) -> None:
        """
        Plot the function's 3D surface in an interactive window.

        Parameters
        ----------
        dims : list, tuple
            The selected dimensions indices to be drawn.
        n_points : int
            Using total n_points**2 points to draw the contour
        font_size:
            Fontsize of subplots title.
        cmap : Union[str, Cmaps]
            The cmap of matplotlib.pyplot.contourf function, use Cmaps for easy selection.
        color:
            Use to make each mesh have a single solid color. ``cmap`` will be overridden
            if color is specified. Either a string, RGB list, or hex color string. For
            example: ``color='white'``, ``color=StrColors.white``, ``color='w'``,
            ``color=[1.0,1.0,1.0]``, or ``color='#FFFFFF'``.
        show_grid :
            Show coordinate plane.
        scale_mode:
            if pass 'xy', normalize both the decision space and objective space to (0, 1).
            if pass 'y', scale objective value to match the decision space, which will keep
            the original decision space information.
        fixed : float, np.ndarray
            Fixed values for all the unused dimensions, if pass a ndarray, its shape should be (dim,)
        plotter:
            The pyvista.Plotter object. If the plotter is not None, plot on it and return, else plot
            on a new created plotter and show.
        """
        show = plotter is None
        try:
            import pyvista as pv
        except ImportError:
            print(f"Please install {colored('pyvista', 'red')} to use interactive plotting.")
            return
        if plotter is None:
            plotter = pv.Plotter(shape=(1, 1))
            plotter.subplot(0, 0)
        x, y, z = self.gen_plot_data(dims=dims, fixed=fixed, n_points=n_points, scale_mode=scale_mode)
        grid = pv.StructuredGrid(x, y, z)
        if color is None:
            plotter.add_mesh(grid, scalars=grid.points[:, 2], cmap=str(cmap))
            plotter.remove_scalar_bar()
        else:
            _color = str(color) if isinstance(color, StrColors) else color
            plotter.add_mesh(grid, color=_color)
        plotter.add_title(f"T{self.id} {self.name}", font_size=font_size)
        if show_grid:
            plotter.show_grid(
                xtitle=f'X{min(dims)}',
                ytitle=f'X{max(dims)}',
                ztitle='Y',
                font_size=8,
                location='outer'
            )
        plotter.add_axes()
        if show:
            plotter.show()

    def set_transform(self, rotation: Optional[np.ndarray]=None, shift: Union[int, float, np.ndarray, None]=None):
        self._transformer.set_transform(rotation, shift)

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
            x_init = x_init * (self.ub - self.lb) + self.lb
        else:
            x_init = x_init * (self._partition[1] - self._partition[0]) + self._partition[0]

        y_init = np.array(self.evaluate(x_init))
        self.solutions.clear()  # Do not remove this line

        self.solutions._x_global = self.x_global
        if not self.auto_update_solutions:
            self.solutions.append(x_init, y_init)
            self.solutions._y_global = self.y_global
        else:
            # because the y_global is getting by self.evaluate(), so
            # we need disable auto_update_solutions temporarily if
            # it is True
            self.auto_update_solutions = False
            self.solutions._y_global = self.y_global
            self.auto_update_solutions = True

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
            return np.random.uniform(low=self.lb, high=self.ub, size=(size, self.dim))

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
        if not np.all(self.lb <= points) or not np.all(points <= self.ub):
            points = np.clip(points, self.lb, self.ub)
        return (points - self.lb) / (self.ub - self.lb)

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
        return points * (self.ub - self.lb) + self.lb

    def transform_x(self, x):
        return self._transformer.transform_x(x)

    def inverse_transform_x(self, x):
        return self._transformer.inverse_transform_x(x)

    def evaluate(self, x: np.ndarray):
        _x = self.before_eval(x)
        y = np.apply_along_axis(self._eval_single, 1, _x)
        y = self.after_eval(x, y)
        return y

    def before_eval(self, x):
        _x = copy.deepcopy(x)
        _x = check_x(_x, self.dim)
        _x = self.transform_x(_x)
        _x = np.clip(_x, self.lb, self.ub)
        return _x.reshape(-1, self.dim)

    def after_eval(self, x, y):
        _x = x.reshape(-1, self.dim)
        _y = y.reshape(-1, self.obj)
        if self.auto_update_solutions:
            self.solutions.append(_x, _y)
        return _y

    @property
    def x_global(self):
        if self._x_global is None:
            return None
        return self.inverse_transform_x(self._x_global)

    @property
    def y_global(self):
        if self.x_global is None:
            return None
        return self.evaluate(self.x_global).squeeze()

    @property
    def shift(self):
        return self._transformer.shift

    @property
    def rotation(self):
        return self._transformer.rotation

    @abstractmethod
    def _eval_single(self, x: np.ndarray):
        """
        OjbFunc:= :math:`\\mathbf{x} \\to f \\to \\mathbf{y}, \\mathbf{x} \\in \\mathbb{R}^{D1},
        \\mathbf{y}\\in \\mathbb{R}^{D2}`, where :math:`D1` is the dimension of decision space
        and :math:`D2` is that of objective space.

        Parameters
        ----------
        x: np.ndarray
            Input vector of the problem.

        Returns
        -------
        np.ndarray or float
        """

    def _check_inputs(self, points):
        return check_x(points, self.dim)

    @property
    def no_partition(self):
        return np.all(self._partition == 0)

    @property
    def id(self) -> int:
        return self._id

    @property
    def lb(self) -> ndarray:
        return self._config.lb

    @property
    def ub(self) -> ndarray:
        return self._config.ub

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def dim(self) -> int:
        return self._config.dim

    @property
    def obj(self):
        return self._config.obj

    @property
    def fe_init(self):
        return self._config.fe_init

    @property
    def fe_max(self):
        return self._config.fe_max

    @property
    def fe_available(self):
        return self.fe_max - self.solutions.size

    @property
    def np_per_dim(self) -> Optional[int]:
        return self._config.np_per_dim

    @property
    def solutions(self) -> Solution:
        return self._solutions


T_Tasks = Union[List[SingleTaskProblem], Tuple[SingleTaskProblem]]


class MultiTaskProblem(ABC):
    is_realworld: bool
    intro = "Not set"
    notes = "Not set"
    references: list[str] = ["Not set"]

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', 123)
        self.random_ctrl = kwargs.pop('random_ctrl', 'weak')
        _init_solutions = kwargs.pop('_init_solutions', True)
        self._problem = self.__init_tasks(*args, **kwargs)
        if _init_solutions:
            self._init_solutions()

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
                raise IndexError(f"Index {index} > total tasks={len(self._problem)} in problem {self.name}")
            return self._problem[index]
        else:
            raise TypeError(f"Index must be an integer or a slice, but {type(index)} given.")

    def _init_solutions(self):
        if self.random_ctrl == 'no':
            self.__init_partition()
            self.__init_solutions()
        elif self.random_ctrl == 'weak':
            self._set_seed()
            self.__init_partition()
            self._unset_seed()
            self.__init_solutions()
        elif self.random_ctrl == 'strong':
            self._set_seed()
            self.__init_partition()
            self.__init_solutions()
            self._unset_seed()
        else:
            raise ValueError('Invalid random_ctrl option.')

    def __init_partition(self):
        for p in self._problem:
            p.init_partition()

    def __init_solutions(self):
        for p in self._problem:
            p.init_solutions()

    def _set_seed(self):
        self._rdm_state = np.random.get_state()
        np.random.seed(self.seed)

    def _unset_seed(self):
        np.random.set_state(self._rdm_state)

    def plot_3d_interactive(
            self,
            tasks_id: Union[list, tuple, range]=(1, 2, 3, 4),
            shape: tuple[int, int]=(2, 2),
            dims=(0, 1),
            font_size: int=10,
            cmap: Union[str, Cmaps]=Cmaps.viridis,
            color: Union[str, StrColors, list[float, float, float], tuple[float, float, float], None]=None,
            n_points: int=100,
            scale_mode: Literal['xy', 'y']='y',
            show_grid: bool=True,
    ):
        """
        Plot tasks in an interactive mode.

        Parameters
        ----------
        tasks_id:
            The tasks id to be plotting.
        shape:
            The grid shape of multi-plots.
        cmap:
            Color map, use `pyfmto.utilities.Cmaps` to easily select alternative options.
        color:
            Use to make each mesh have a single solid color. ``cmap`` will be overridden
            if color is specified. Either a string, RGB list, or hex color string. For
            example: ``color='white'``, ``color=StrColors.white``, ``color='w'``,
            ``color=[1.0,1.0,1.0]``, or ``color='#FFFFFF'``.
        dims:
            The selected dims to be plotting.
        scale_mode:
            if pass 'xy', normalize both the decision space and objective space to (0, 1).
            if pass 'y', scale objective value to match the decision space, which will keep
            the original decision space information.
        n_points:
            Number of points in one dimension, total n_points**2 points.
        font_size:
            Fontsize of subplots title.
        show_grid:
            If show coordinates plane.
        """
        if np.any(np.array(tasks_id) <= 0) or np.any(np.array(tasks_id) > self.task_num):
            raise ValueError(f"tasks id should in range [1, {self.task_num}]")
        if len(tasks_id) > shape[0] * shape[1]:
            raise ValueError(f"The number of clients {len(tasks_id)} is larger than "
                             f"the number of cells in grid {shape[0]}*{shape[1]}")
        try:
            import pyvista as pv
        except ImportError:
            print(f"Please install {colored('pyvista', 'red')} to use this function")
            return
        plotter = pv.Plotter(shape=shape)
        for index, tid in enumerate(tasks_id):
            row, col = index // shape[1], index % shape[1]
            plotter.subplot(row, col)
            task = self._problem[tid-1]
            task.plot_3d_interactive(
                dims=dims,
                n_points=n_points,
                scale_mode=scale_mode,
                cmap=cmap,
                color=color,
                show_grid=show_grid,
                font_size=font_size,
                plotter=plotter,
            )
        plotter.show()

    def plot_distribution(
            self,
            dims=(0, 1),
            style: Union[str, SeabornStyles]=SeabornStyles.whitegrid,
            figsize: tuple[float, float, float]=(11., 9., 1.),
            filename=None
    ):
        """
        Plot the distribution of initial solutions for each task in the multitask problem.

        This function visualizes how the initial solutions are distributed across the
        decision space for each task. It helps to understand the partitioning strategy
        and the diversity of initial samples among different tasks.

        Parameters
        ----------
        dims : tuple of int, default=(0, 1)
            The two dimensions to plot. Only 2D visualization is supported.

        style : str or SeabornStyles, default=SeabornStyles.whitegrid
            The seaborn style to use for the plot. Can be any valid seaborn style
            or a SeabornStyles enum value.

        figsize : tuple of float, default=(11., 9., 1.)
            Figure size specification as (width, height, scale_factor). The final
            figure size will be (width*scale_factor, height*scale_factor).

        filename : str, optional
            Path to save the figure. If None, the figure will be displayed directly.

        Notes
        -----
        - The function normalizes all solution points to [0, 1] space for consistent visualization.
        - Grid lines are shown according to the 'np_per_dim' setting to visualize partitions.
        - Each task is represented with a unique color and marker style.
        - The plot includes a legend indicating which color/marker corresponds to which task.

        Examples
        --------
        >>> problem.plot_distribution()
        >>> problem.plot_distribution(dims=(1, 2), filename='distribution.png')
        """
        init_x = {f"T{p.id:02}({p.name})": p.normalize_x(p.solutions.x) for p in self}
        np_per_dim = self[0].np_per_dim
        tick_interval = 1.0 / np_per_dim
        ticks = np.arange(0, 1 + tick_interval, tick_interval)
        dim1 = min(dims)
        dim2 = max(dims)
        data = []
        for label, points in init_x.items():
            for point in points:
                data.append({
                    'x': point[dim1],
                    'y': point[dim2],
                    'problem': label
                })
        w, h, s = figsize
        _figsize = (w*s, h*s)
        plt.figure(figsize=_figsize)
        sns.set(style=str(style))
        sns.scatterplot(data=pd.DataFrame(data), x='x', y='y', hue='problem', style='problem', s=80)
        plt.title(f'Initial Solutions of Each Task (np_per_dim={np_per_dim})')
        plt.xlabel(f'X{dim1}')
        plt.ylabel(f'X{dim2}', rotation=0)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.legend(title=f'Tasks', bbox_to_anchor=(1., 1), loc='upper left')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_heatmap(
            self,
            n_samples: int=1000,
            method: Literal['kendalltau', 'spearmanr', 'pearsonr']='spearmanr',
            p_ub: float=0.05,
            figsize: tuple[float, float, float]=(9., 7., 1.),
            cmap: Optional[Union[str, Cmaps]]=None,
            masker: Optional[float]=np.nan,
            font_size: int=10,
            fmt: str= '.2f',
            triu: Literal['full', 'lower', 'upper']='full',
            vmax: Optional[float]=None,
            vmin: Optional[float]=None,
            linewidth: float=0.5,
            filename=None,
    ):
        """
        Plot a heatmap showing the correlation between tasks in the multitask problem.

        This function computes correlation coefficients between all pairs of tasks
        using sampled points in the decision space, and visualizes them as a heatmap.
        Correlations that are not statistically significant (based on p-value) can
        be masked out.

        Parameters
        ----------
        n_samples : int, default=1000
         Number of Latin Hypercube samples to use for evaluating the functions.

        method : {'spearmanr', 'kendalltau', 'pearsonr'}, default='spearmanr'
         Correlation method to use:
         - 'spearmanr': Spearman rank correlation
         - 'kendalltau': Kendall tau correlation
         - 'pearsonr': Pearson correlation coefficient

        p_ub : float, default=0.05
         Upper bound for p-value. Correlations with p-values greater than this
         value are considered not statistically significant and will be masked.

        figsize : tuple[float, float, float], default=(9., 7., 1.)
         Figure size as (width, height, scale).

        cmap : str or Cmaps, optional
         Colormap for the heatmap. If None, uses the default matplotlib colormap.

        masker : float, optional
         Value to use for masking non-significant correlations. Default is NaN.

        font_size : int, default=10
         Font size for annotation text in heatmap cells.

        fmt : str, default='.2f'
         Format string for annotating values in heatmap cells.

        triu : {'full', 'lower', 'upper'}, default='full'
         Which part of the matrix to display:
         - 'full': Show full matrix
         - 'lower': Show only lower triangular matrix
         - 'upper': Show only upper triangular matrix

        vmax : float, optional
         Maximum value for colormap normalization.

        vmin : float, optional
         Minimum value for colormap normalization.

        linewidth : float, default=0.5
         Width of lines separating heatmap cells.

        filename : str, optional
         File path to save the plot. If None, displays the plot instead.

        Notes
        -----
        The heatmap shows correlation coefficients between tasks. Non-significant
        correlations (based on the p_ub threshold) are masked with the masker value.
        The diagonal always shows 1.0 as each task is perfectly correlated with itself.
        Task labels are formatted as 'T{ID:02}' where ID is the task identifier.

        See Also
        --------
        See `scipy.stats.kendalltau`, `spearmanr`, or `pearsonr` documentation for more detail.
        """
        methods = {
            'spearmanr': spearmanr,
            'kendalltau': kendalltau,
            'pearsonr': pearsonr,
        }
        if method in methods:
            corr = methods[method]
        else:
            raise ValueError(f"Invalid method: {method}, support spearmanr, kendalltau, and pearsonr")
        _cmap = None if cmap is None else str(cmap)
        x = lhs(self[0].dim, samples=n_samples)
        evals = [f.evaluate(f.denormalize_x(x)).squeeze() for f in self]
        evals = np.array(evals)
        significance = [[[*corr(evals[i], evals[j])] for j in range(self.task_num)] for i in range(self.task_num)]
        significance = np.array(significance)
        statis = significance[:, :, 0]
        pvalue = significance[:, :, 1]
        triu_mask = np.triu(np.ones_like(statis, dtype=bool), k=1)
        statis[pvalue > p_ub] = masker if masker else np.nan
        if triu == 'lower':
            statis[~triu_mask] = np.nan
        elif triu == 'upper':
            statis[triu_mask] = np.nan
        cols = [f"T{f.id:02}" for f in self]
        w, h, s = figsize
        _figsize = (w*s, h*s)
        df = pd.DataFrame(statis[::-1], columns=cols, index=cols[::-1])
        _, ax = plt.subplots(figsize=_figsize)
        sns.heatmap(
            data=df,
            cmap=_cmap,
            annot=True,
            fmt=fmt,
            linewidths=linewidth,
            ax=ax,
            annot_kws={"size": font_size},
            vmax=vmax,
            vmin=vmin,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def task_num(self):
        return len(self._problem)

    def get_info(self) -> dict:
        return {
            "ProbName": [self.name],
            "ProbType": ['Realworld' if self.is_realworld else 'Synthetic'],
            "TaskNum": [len(self._problem)],
            "DecDim": [self._problem[0].dim],
            "ObjDim": [self._problem[0].obj]
        }

    def __len__(self):
        return len(self._problem) if self._problem is not None else 0

    def __repr__(self):
        t = "realworld" if self.is_realworld else "synthetic"
        return f"{self.name}({len(self._problem)} {t} tasks)"

    def __str__(self):
        def _reformat_lines(text, width=80):
            _dedent = textwrap.dedent(text)
            _width = textwrap.fill(_dedent, width=width)
            _indent = textwrap.indent(_width, ' ' * 4)
            if not _indent.startswith('\n'):
                _indent = '\n' + _indent
            if not _indent.endswith('\n'):
                _indent += '\n'
            return _indent

        tab = tabulate(self.get_info(), headers="keys", tablefmt="rounded_grid")
        tab = textwrap.indent(tab, " " * 8)
        tab_width = tab.find('\n') + 8
        prob_name = self.name.center(tab_width-16, '=')
        prob_name  = textwrap.indent(prob_name, " " * 8)
        intro = _reformat_lines(self.intro, tab_width)
        notes = _reformat_lines(self.notes, tab_width)
        ref = [_reformat_lines(txt, tab_width) for txt in self.references]
        ref = "\n".join(ref)

        return (
            f"{prob_name}\n"
            f"{tab}\n"
            f"Introduction:{intro}\n"
            f"Notes:{notes}\n"
            f"References:{ref}"
        ).replace('\n\n\n', '\n\n')
