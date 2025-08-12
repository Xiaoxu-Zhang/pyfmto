import inspect
import pandas as pd
from collections import defaultdict
from tabulate import tabulate

from .problem import SingleTaskProblem, MultiTaskProblem
from .solution import Solution
from . import realworld
from . import synthetic

__all__ = [
    'load_problem',
    'list_problems',
    'MultiTaskProblem',
    'Solution',
    'SingleTaskProblem'
]


def collect_problems_meta():
    results = {}
    for module in [realworld, synthetic]:
        for name in dir(module):
            cls = getattr(module, name)
            if inspect.isclass(cls) and issubclass(cls, MultiTaskProblem) and cls != MultiTaskProblem:
                results[name] = cls
    return results


PROBLEMS = collect_problems_meta()
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
        if issubclass(cls, MultiTaskProblem):
            instance = cls(_init_solutions=False)
            prob_type = "Realworld" if instance.is_realworld else "Synthetic"
            data['name'].append(name)
            data['total'].append(instance.task_num)
            data['type'].append(prob_type)
            data['dim'].append(instance[0].dim)
            data['obj'].append(instance[0].obj)
    df = pd.DataFrame(data)
    if print_it:
        print(tabulate(df, headers='keys', tablefmt='rounded_grid', showindex=False))
    return list(PROBLEMS.keys())
