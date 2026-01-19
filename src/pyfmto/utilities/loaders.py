import importlib
import inspect
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import no_type_check

import tabulate

from pyfmto.core.typing import TComponent, TComponentList, TComponentNames, TDiscoverResult

from ..framework import AlgorithmData
from ..problem import ProblemData
from .loggers import logger
from .tools import add_sources, print_dict_as_table

__all__ = [
    'discover',
    'list_algorithms',
    'list_components',
    'list_problems',
    'load_algorithm',
    'load_component',
    'load_problem',
]

_DISCOVER_CACHE: TDiscoverResult = {}


@no_type_check
def load_algorithm(name: str, sources: list[str], **kwargs) -> AlgorithmData:
    return load_component('algorithms', name, sources, **kwargs)


@no_type_check
def load_problem(name: str, sources: list[str], **kwargs) -> ProblemData:
    return load_component('problems', name, sources, **kwargs)


def list_problems(sources: list[str], print_it=False) -> dict[str, list[str]]:
    return list_components('problems', sources, print_it)


def list_algorithms(sources: list[str], print_it=False) -> dict[str, list[str]]:
    return list_components('algorithms', sources, print_it)


def list_components(
        target: TComponentNames,
        sources: list[str],
        print_it=False
) -> dict[str, list]:
    components = discover(sources).get(target, {})
    res: dict[str, list] = defaultdict(list)
    for _, comp_lst in components.items():
        for comp in comp_lst:
            for key, val in comp.desc.items():
                res[key].append(val)
    src_str = textwrap.indent('\n'.join(sources), '  ')
    if len(res.get('name', [])) == 0:
        summary = f"No {target} found in:\n{src_str}"
        logger.warning(summary)
    else:
        n_available = sum(res.get('available', []))
        n_total = len(res.get('name', []))
        summary = f"Found {n_available} available (total {n_total}) {target} in:\n{src_str}"
        logger.debug(f"{summary}\n{tabulate.tabulate(res, headers='keys', tablefmt='rounded_grid')}")
    if print_it:
        print(summary)
        print_dict_as_table(res)
    return res


@no_type_check
def load_component(
        target: TComponentNames,
        name: str,
        sources: list[str],
        **kwargs
) -> TComponent:

    components = discover(sources).get(target, {})
    comp = _empty_data(target)
    comp.name_orig = name
    for comp in components.get(name, []):
        if comp.available:
            comp.params_update = kwargs
            return comp
    return comp


@no_type_check
def discover(paths: list[str]) -> TDiscoverResult:
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


@no_type_check
def _find_components(subdir: Path) -> dict[str, TComponentList]:
    results: dict[str, TComponentList] = defaultdict(list)
    try:
        module = importlib.import_module('.'.join(subdir.parts[-3:]))
        for attr_name in dir(module):
            if attr_name.startswith('__'):
                continue
            attr = getattr(module, attr_name)
            if attr in [AlgorithmData, ProblemData]:
                continue
            if inspect.isclass(attr) and issubclass(attr, (AlgorithmData, ProblemData)):
                obj = attr()
                obj.source = str(subdir)
                results[obj.name_orig].append(obj)
    except Exception as e:
        obj = _empty_data(str(subdir))
        obj.name_orig = subdir.name
        obj.source = str(subdir)
        obj.issues = [str(e)]
        results[obj.name_orig].append(obj)
        logger.warning(f"Failed to load '{subdir}': {e!s}")

    return results


def _empty_data(target: str):
    return AlgorithmData() if 'algorithm' in target else ProblemData()

