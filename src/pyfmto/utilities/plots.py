from typing import Optional, Callable, Union
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyfmto.utilities.stroptions import Cmaps


def gen_func_data(func, dim, dims, lb, ub, n_points, fixed=None):
    dim1, dim2 = sorted(dims)
    if not dim1 in range(dim) or not dim2 in range(dim):
        raise ValueError(f"The selected_dims's values should in [0, {dim - 1}].")

    d1 = np.linspace(lb[dim1], ub[dim1], n_points)
    d2 = np.linspace(lb[dim2], ub[dim2], n_points)
    D1, D2 = np.meshgrid(d1, d2)

    fix_to = (lb + ub) / 2 if fixed is None else fixed
    points = np.ones(shape=(n_points, n_points, dim)) * fix_to
    points[:, :, dim1] = D1
    points[:, :, dim2] = D2

    Z = np.apply_along_axis(func, axis=-1, arr=points)
    Z = Z.squeeze()

    return D1, D2, Z


def plot_func_2d(
        func: Callable[[np.ndarray], np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        dims: tuple[int, int]=(0, 1),
        n_points: int=100,
        cmap: Union[str, Cmaps]=Cmaps.viridis,
        levels: int=30,
        alpha: float=0.7,
        fixed: Union[float, np.ndarray, None]=None,
        ax: Optional[plt.Axes]=None,
) -> plt.Axes:
    """
    Plot the function 2D contour.

    Parameters
    ----------
    func : callable
        The callable function that is used to calculate the value
    lb : np.ndarray
        The lower bound of the variables.
    ub : np.ndarray
        The upper bound of the variables.
    dim: int
        The input dimension of the function
    dims : list, tuple
        The selected dimensions indices to be drawn.
    n_points : int
        Using total n_points**2 points to draw the contour
    cmap : str
        The cmap of matplotlib.pyplot.contourf function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)
    levels : int
        The levels parameter of contourf function
    alpha : float
        The alpha parameter of contourf function
    fixed : float, np.ndarray
        Fixed values for all the unused dimensions, if pass a ndarray, its shape should be (dim,)
    ax: plt.Axis
        Axes object to plot on. If None, uses current axes.
    """
    if not isinstance(dims, (list, tuple)):
        raise ValueError('selected_dims should be [int, int] or (int, int)')
    D1, D2, Z = gen_func_data(func, dim, dims, lb, ub, n_points, fixed)
    ax = plt.gca() if ax is None else ax
    fig = ax.get_figure()
    cont = ax.contourf(D1, D2, Z, levels=levels, cmap=str(cmap), alpha=alpha)
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Function Value')
    cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.2e}' if abs(x) > 1e4 else f'{x:.2f}')
    cbar.update_ticks()

    return ax


def plot_func_3d(
        func: Callable[[np.ndarray], np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        dims: tuple[int, int] = (0, 1),
        n_points: int = 100,
        cmap: Union[str, Cmaps]=Cmaps.viridis,
        alpha: float = 0.7,
        levels: int = 30,
        fixed: Union[float, np.ndarray, None] = None,
        ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the function 3D surface.

    Parameters
    ----------
    func : callable
        The callable function that is used to calculate the value
    lb : np.ndarray
        The lower bound of the variables.
    ub : np.ndarray
        The upper bound of the variables.
    dim: int
        The input dimension of the function
    dims : tuple
        The selected dimensions indices to be drawn.
    n_points : int
        Using total n_points**2 points to draw the surface
    cmap : str
        The colormap for the surface plot
    levels: int
        The levels parameter of contourf function
    alpha : float
        The alpha parameter of surface function
    fixed : float, np.ndarray
        Fixed values for all the unused dimensions, if pass a ndarray, its shape should be (dim,)
    ax : plt.Axes, optional
        Axes object to plot on. If None, uses current axes.

    Returns
    -------
    plt.Axes
        The axes object with the plot
    """
    if not isinstance(dims, (list, tuple)):
        raise ValueError('dims should be [int, int] or (int, int)')
    D1, D2, Z = gen_func_data(func, dim, dims, lb, ub, n_points, fixed)
    ax = plt.gca() if ax is None else ax
    fig = ax.get_figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D1, D2, Z, cmap=str(cmap), edgecolor='none', alpha=alpha)
    ax.contour(D1, D2, Z, zdir='z', offset=np.min(Z), cmap=str(cmap), levels=levels)
    cbar = fig.colorbar(surf, ax=ax)
    cbar.set_label('Function Value')
    cbar.formatter = FuncFormatter(lambda x, pos: f'{x:.2e}' if abs(x) > 1e4 else f'{x:.2f}')
    cbar.update_ticks()

    return ax
