import inspect
import pandas as pd
from collections import defaultdict
from tabulate import tabulate

from .problem import *
from .solution import *
from .realworld import *
from .synthetic import *

__omit__ = [
    'inspect',
    'pd',
    'defaultdict',
    'tabulate'
]

__all__ = [
    'PROBLEMS',
    'load_problem',
    'list_problems',
    'MultiTaskProblem',
    'Solution',
    'SingleTaskProblem',
    'check_and_transform'
]

PROBLEMS = {name: cls
    for name, cls in globals().items()
    if inspect.isclass(cls) and name not in __omit__ + __all__
}

_lowercase_map = {name.lower(): name for name in PROBLEMS}


def load_problem(name, **kwargs) -> MultiTaskProblem:
    no_space = name.replace('_', '')
    if name in PROBLEMS:
        return PROBLEMS[name](**kwargs)
    elif no_space in _lowercase_map:
        return PROBLEMS[_lowercase_map[no_space]](**kwargs)
    else:
        raise ValueError(f"Problem '{name}' not found, call list_problems() to see available problems.")


def list_problems(print_it=False):
    data = defaultdict(list)
    for name, cls in PROBLEMS.items():
        instance: MultiTaskProblem = cls(_init_solutions=False)
        data['name'].append(name)
        data['total'].append(instance.task_num)
        prob_type = "Realworld" if instance.is_realworld else "Synthetic"
        data['type'].append(prob_type)
        data['dim'].append(instance[0].dim)
        data['obj'].append(instance[0].obj)
    df = pd.DataFrame(data)
    if print_it:
        print(tabulate(df, headers='keys', tablefmt='rounded_grid', showindex=False))
    return list(PROBLEMS.keys())
