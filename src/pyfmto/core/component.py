import copy
import inspect
from typing import Any

from deepdiff import DeepDiff

from ..utilities.io import dumps_yaml, parse_yaml, recursive_to_pure_dict
from ..utilities.tools import deepmerge


class ComponentData:

    def __init__(self):
        self.name_orig: str = ''
        self.name_alias: str = ''
        self.params_default: dict[str, Any] = {}
        self.params_update: dict[str, Any] = {}
        self.source: str = ''
        self.issues: list[str] = []

        if not self.name_orig:
            self.name_orig = self.__class__.__name__
        if self.available:
            self.load_default_params()

    @property
    def desc(self) -> dict:
        return {
            'name': self.name_orig,
            'alias': self.name_alias,
            'available': self.available,
            'issues': '; '.join(self.issues),
            'source': self.source,
        }

    def __str__(self):
        return '\n'.join([f"{k}: {v}" for k, v in self.desc.items()])

    def __repr__(self):
        info_lst = [
            f"{type(self).__name__}({self.name})",
            f"  available({self.available})",
            f"  orig({self.name_orig})",
            f"  alias({self.name_alias})",
        ]
        return '\n'.join(info_lst)

    @property
    def available(self) -> bool:
        return False

    def load_default_params(self) -> None:
        raise NotImplementedError

    @property
    def params(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def params_snapshot(self) -> str:
        data = {
            self.name: {
                'base': self.name_orig,
                'params': self.params,
                'default': self.params_default,
                'update': self.params_update,
            }
        }
        return dumps_yaml(recursive_to_pure_dict(data))

    @property
    def name(self) -> str:
        return self.name_alias if self.name_alias else self.name_orig

    @property
    def params_yaml(self) -> str:
        if not self.available:
            return f"{self.name} is not available"
        prefix = "Problem" if hasattr(self, 'problem') else "Algorithm"
        if self.params_default:
            return dumps_yaml(self.params_default)
        else:
            return f"{prefix} '{self.name}' has no configurable parameters."

    @property
    def params_diff(self) -> str:
        return DeepDiff(self.params_default, self.params).pretty()

    def _merged_params(self):
        return deepmerge(
            copy.deepcopy(self.params_default),
            copy.deepcopy(self.params_update),
        )

    def _parse_default_params(self, attr_name: str):
        try:
            return parse_yaml(getattr(self, attr_name).__doc__)
        except Exception:
            return {}

    def _check_attr(self, name: str, cls: type):
        attr = getattr(self, name, None)
        not_none = attr is not None
        cls_ok = inspect.isclass(attr) and issubclass(attr, cls)
        not_abstract = not inspect.isabstract(attr)
        return not_none and cls_ok and not_abstract
