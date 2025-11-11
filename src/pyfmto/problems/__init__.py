import inspect
import pandas as pd
from collections import defaultdict
from importlib import import_module
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
    problems = {}
    modules = [realworld, synthetic]
    try:
        user_defined = import_module('problems')
        modules.append(user_defined)
    except ImportError:
        pass

    for module in modules:
        for name in dir(module):
            cls = getattr(module, name)
            if inspect.isclass(cls) and issubclass(cls, MultiTaskProblem) and cls != MultiTaskProblem:
                problems[name] = cls
    lowercase_name = {name.lower(): name for name in problems}
    return problems, lowercase_name


def load_problem(name, **kwargs) -> MultiTaskProblem:
    problems, lowercase_name = collect_problems_meta()
    no_space = name.replace('_', '').lower()
    if name in problems:
        return problems[name](**kwargs)
    elif no_space in lowercase_name:
        return problems[lowercase_name[no_space]](**kwargs)
    else:
        raise ValueError(f"Problem '{name}' not found, call list_problems() to see available problems.")


def list_problems(print_it=False):
    problems, _ = collect_problems_meta()
    data = defaultdict(list)
    for name, cls in problems.items():
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
    return list(problems.keys())
