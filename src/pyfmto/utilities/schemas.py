import numpy as np
import warnings
from typing import Union, Optional, cast
from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from pyfmto.utilities import show_in_table
from pyfmto.utilities.confparser import ExperimentConfig

T_Bound = Union[int, float, list, tuple, np.ndarray]


class STPConfig(BaseModel):
    """
    Basically, this model makes the following validation and convertion:

    - ``dim``: a positive integer
    - ``obj``: a positive integer
    - ``lb``: finally a ndarray with shape (dim,)
    - ``ub``: finally a ndarray with shape (dim,)
    - ``fe_init``: a positive integer, default to ``5*dim``
    - ``fe_max``: a positive integer, default to ``11*dim``
    - ``np_per_dim``: a positive integer, default to ``1``

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
    fe_init: int = -1
    fe_max: int = -1
    np_per_dim: int = 1

    @field_validator('dim', 'obj')
    def validate_positive_integer(cls, v):
        if v <= 0:
            raise ValueError('dim and obj must be positive integers')
        return v

    @field_validator('fe_init', 'fe_max', 'np_per_dim')
    def validate_positive_or_none(cls, v):
        if v < 1:
            raise ValueError(f"Invalid value: {v} (fe_init, fe_max, and np_per_dim must be positive)")
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

        self.lb = cast(np.ndarray, self.lb)
        self.ub = cast(np.ndarray, self.ub)

        # Check dimensionality
        if self.lb.shape != (self.dim,):
            raise ValueError(f"lb must be a scalar or array of shape ({self.dim},)")
        if self.ub.shape != (self.dim,):
            raise ValueError(f"ub must be a scalar or array of shape ({self.dim},)")

        # Check bounds
        if not np.all(self.lb < self.ub):
            raise ValueError("All elements of lb must be less than corresponding elements of ub")

        # Set default values for fe_init and fe_max
        if self.fe_init == -1:
            self.fe_init = 5 * self.dim
        if self.fe_max == -1:
            self.fe_max = 11 * self.dim

        if self.fe_init > self.fe_max:
            raise ValueError("fe_init must be less than or equal to fe_max")

        return self


class TransformerConfig(BaseModel):
    """
    Basically, this model makes the following validation and conversion:

    - ``dim``: a positive integer (has been validated by STPConfig)
    - ``shift``: finally a ndarray with shape (dim, )
    - ``rotation``: finally a ndarray with shape (dim, dim)
    - ``rotation_inv``: set to inverse of rotation matrix
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    dim: int
    shift: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    rotation_inv: Optional[np.ndarray] = None

    @model_validator(mode='after')
    def check_rotation_and_shift(self):
        if self.rotation is None:
            self.rotation = np.eye(self.dim)
            self.rotation_inv = np.eye(self.dim)
        if self.shift is None:
            self.shift = np.zeros(self.dim)

        if self.rotation.shape != (self.dim, self.dim):
            raise ValueError(f"{self.dim} dimensional task's rotation shape must be ({self.dim}, {self.dim})")
        if self.shift.shape != (self.dim,):
            raise ValueError(f"{self.dim} dimensional task's shift shape must be ({self.dim},)")
        self.rotation = self.rotation.T
        self.rotation_inv = np.linalg.inv(self.rotation.T)

        return self


class FunctionInputs(BaseModel):
    """
    Basically, this model makes the following validation and reshaping:

    - ``x``: finally a ndarray with shape (n, dim)
    - ``dim``: a positive integer (has been validated by STPConfig)

    Notes
    -----
    Additionally, this model validates that:
        1. ``x`` must be a numpy array.
        2. The final shape of ``x`` must be (n, dim).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    x: np.ndarray
    dim: int

    @model_validator(mode='after')
    def check_and_reshape_x(self):
        if self.x.ndim <= 1:
            self.x = self.x.reshape(-1, self.dim)
        if self.x.ndim != 2:
            raise ValueError(f'x must have shape (n, {self.dim}), got {self.x.shape} instead')
        if self.x.shape[1] != self.dim:
            raise ValueError(f'x must have shape (n, {self.dim}), got {self.x.shape} instead')
        return self


class LauncherConfig(BaseModel):
    results: str = 'out/results'
    repeat: int = 1
    seed: int = 42
    backup: bool = True
    save: bool = True
    loglevel: str = 'INFO'
    algorithms: list[str]
    problems: list[str]
    experiments: list[ExperimentConfig] = []

    @field_validator('results', mode='before')
    def results_must_be_not_none(cls, v):
        if not isinstance(v, (str, type(None))):
            raise TypeError(f'results must be a string or None, got {type(v)} instead')
        return v if v is not None else 'out/results'

    @field_validator('loglevel')
    def loglevel_must_be_valid(cls, v):
        valid_values = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_values:
            raise ValueError(f'loglevel must be one of {valid_values}, got {v} instead')
        return v

    @field_validator('repeat', 'seed')
    def integer_must_be_positive(cls, v):
        if v < 1:
            raise ValueError('repeat must be >= 1')
        return v

    @field_validator('algorithms', 'problems')
    def lists_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('list must have at least 1 element')
        return v

    def show_summary(self):
        colored_tab, _ = show_in_table(
            num_exp=len(self.experiments),
            repeat_per_exp=self.repeat,
            total_repeat=self.n_exp * self.repeat,
        )
        print(colored_tab)

    @property
    def n_exp(self) -> int:
        return len(self.experiments)


class ReporterConfig(BaseModel):
    results: Optional[str] = 'out/results'
    algorithms: list[list[str]]
    problems: list[str]

    @field_validator('results')
    def results_must_be_not_none(cls, v):
        return v if v is not None else 'out/results'

    @field_validator('algorithms')
    def inner_lists_must_have_min_length(cls, v):
        for inner_list in v:
            if len(inner_list) < 1:
                raise ValueError('inner lists must have at least 1 elements')
        return v

    @field_validator('problems', 'algorithms')
    def outer_list_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('problems list must have at least 1 element')
        return v


class PlottingArgs(BaseModel):
    """
    This model will make sure:

    - ``dim:int>=2``
    - ``0<=dims[0]<dims[1]<dim``
    - ``10<=n_points:int<=1000``
    - ``lb<=fixed:ndarray<=ub``, ``fixed.shape=(dim,)``
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dim: int
    dims: tuple
    n_points: int
    lb: np.ndarray
    ub: np.ndarray
    fixed: Union[int, float, np.ndarray, None]

    @field_validator('dim')
    def dim_must_be_ge_2(cls, v):
        if v < 2:
            raise ValueError(f"only support dim>=2 functions, got dim={v}")
        return v

    @field_validator('dims')
    def dims_must_ordered_and_different(cls, v):
        if len(v) != 2:
            raise ValueError(f"dims must be a tuple of two >=0 integers, got {v}")
        if min(v) < 0:
            raise ValueError("dims must be integers and >= 0")
        if v[0] == v[1]:
            raise ValueError(f"dims should be two different integers, got dims={v}")
        return min(v), max(v)

    @field_validator('n_points')
    def warning_if_too_large(cls, v):
        if v > 1000:
            warnings.warn("A large n_points may cause slow plotting. Using 1000 instead.")
            return 1000
        if v < 10:
            warnings.warn("A small n_points may cause detail loss in plotting. Using 10 instead.")
            return 10
        return v

    @model_validator(mode='after')
    def check_and_set_defaults(self):
        if self.dims[1] >= self.dim:
            raise ValueError(f"selected dim must be in [0, {self.dim - 1}], got dims={self.dims}")
        if self.fixed is None:
            self.fixed = (self.lb + self.ub) / 2
        elif isinstance(self.fixed, (int, float)):
            self.fixed = np.ones(self.dim) * self.fixed
        self.fixed = cast(np.ndarray, self.fixed)
        if self.fixed.shape != (self.dim,):
            raise ValueError(f"fixed shape, if a ndarray, must be ({self.dim},), got fixed shape={self.fixed.shape}")
        if np.any(self.fixed < self.lb) or np.any(self.fixed > self.ub):
            raise ValueError(f"fixed point must be within [lb, ub], got \n"
                             f"fixed={self.fixed}\nlb={self.lb}\nub={self.ub}")
        return self
