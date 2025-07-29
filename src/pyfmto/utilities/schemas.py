import numpy as np
from typing import Union, Optional
from pydantic import BaseModel, field_validator, model_validator, ConfigDict, StrictInt, StrictFloat

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
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


class FunctionInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    x: Union[StrictInt, StrictFloat, list, tuple, np.ndarray]
    dim: int

    @field_validator('x')
    def x_must_ndarray(cls, v):
        if isinstance(v, (int, float)):
            return np.array([v])
        if isinstance(v, (list, tuple)):
            return np.array(v)
        return v

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
    results: Optional[str] = 'out/results'
    repeat: int = 1
    seed: int = 42
    backup: bool = True
    save: bool = True
    algorithms: list[str]
    problems: list[str]

    @field_validator('results')
    def results_must_be_not_none(cls, v):
        return v if v is not None else 'out/results'

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
            if len(inner_list) < 2:
                raise ValueError('inner lists must have at least 2 elements')
        return v

    @field_validator('problems', 'algorithms')
    def outer_list_must_not_be_empty(cls, v):
        if len(v) < 1:
            raise ValueError('problems list must have at least 1 element')
        return v
