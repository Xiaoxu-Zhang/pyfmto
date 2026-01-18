import importlib
import inspect
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pyfmto.framework import AlgorithmData
from pyfmto.utilities.tools import print_dict_as_table

from ..core import ComponentData
from ..problem import MultiTaskProblem, ProblemData
from .loggers import logger
from .tools import add_sources

__all__ = [
    'load_problem',
    'discover',
]

_DISCOVER_CACHE: dict[str, dict[str, list[ComponentData]]] = {}


def load_problem(name: str, sources: list[str], **kwargs) -> MultiTaskProblem:
    problems = discover(sources).get('problems', {})
    prob = problems.get(name, ProblemData())
    if not prob.available:
        raise ValueError(f"Problem '{name}' is not available")
    prob.params_update = kwargs
    return prob.initialize()


def list_problems(sources: list[str], print_it=False) -> dict[str, list[str]]:
    return list_components('problems', sources, print_it)


def list_algorithms(sources: list[str], print_it=False) -> dict[str, list[str]]:
    return list_components('algorithms', sources, print_it)


def list_components(
        target: Literal['algorithms', 'problems'],
        sources: list[str],
        print_it=False
) -> dict[str, list[str]]:
    components = discover(sources).get(target, {})
    res: dict[str, list[str]] = defaultdict(list)
    for _, comp_lst in components.items():
        for comp in comp_lst:
            for key, val in comp.desc.items():
                res[key].append(val)
    if print_it:
        print_dict_as_table(res)
    return res


def discover(paths: list[str]) -> dict[str, dict[str, list[ComponentData]]]:
    global _DISCOVER_CACHE
    if _DISCOVER_CACHE:
        return _DISCOVER_CACHE

    _DISCOVER_CACHE = {
        'algorithms': defaultdict(list),
        'problems': defaultdict(list),
    }
    add_sources(paths)
    for target in ['algorithms', 'problems']:
        for path in paths:
            target_dir = Path(path).resolve() / target
            if target_dir.exists():
                for name in os.listdir(target_dir):
                    sub_dir = target_dir / name
                    if sub_dir.is_dir() and not name.startswith(('.', '_')):
                        for key, res in _find_components(sub_dir).items():
                            _DISCOVER_CACHE[target][key].extend(res)
            else:
                logger.warning(f'{target_dir} does not exist')
    return _DISCOVER_CACHE


def _find_components(subdir: Path):
    results: dict[str, list[ComponentData]] = defaultdict(list)
    try:
        module = importlib.import_module('.'.join(subdir.parts[-3:]))
        for attr_name in dir(module):
            if attr_name.startswith('__'):
                continue
            attr = getattr(module, attr_name)
            if attr in [AlgorithmData, ProblemData]:
                continue
            if inspect.isclass(attr) and issubclass(attr, ComponentData):
                obj = attr()
                obj.source = str(subdir)
                results[obj.name_orig].append(obj)
    except Exception as e:
        obj = ComponentData()
        obj.name_orig = subdir.name
        obj.source = str(subdir)
        obj.issues = [str(e)]
        results[obj.name_orig].append(obj)
        logger.warning(f"Failed to load '{subdir}': {e!s}")

    return results
