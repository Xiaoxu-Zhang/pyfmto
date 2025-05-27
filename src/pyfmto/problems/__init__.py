
import yaml
from pathlib import Path
from typing import Type, Optional

from .problem import *
from .solution import *
from .realworld import *
from .synthetic import *

__all__ = ['load_problem', 'SingleTaskProblem', 'MultiTaskProblem', 'Solution', 'check_and_transform']


def load_problem(problem_name, **kwargs):
    cls_name, default_kwargs = _retriv_problem(problem_name)
    cls: Type[MultiTaskProblem] = globals()[cls_name]
    default_kwargs.update(kwargs)
    return cls(**default_kwargs), default_kwargs


def _retriv_problem(prob_name: str) -> Optional[tuple[str, dict]]:
    conf_file = Path(__file__).parent / 'problems.yaml'
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)
    for problem_type, sub_types in conf.items():
        for sub_type, problems in sub_types.items():
            if prob_name in problems:
                cls_name = problems[prob_name]['class']
                args = problems[prob_name].get('args', {})
                return cls_name, args
    raise ValueError(f"Problem '{prob_name}' not found, see README for available problems.")
