import shutil
import sys
import unittest
from pathlib import Path
from typing import Union

from .generators import gen_code, ExpDataGenerator
from .launchers import running_server, start_clients

__all__ = [
    "PyfmtoTestCase",
    "ExpDataGenerator",
    "running_server",
    "start_clients",
    "gen_code",
]


class PyfmtoTestCase(unittest.TestCase):
    def save_sys_env(self):
        self._sys_paths = list(sys.path)
        self._sys_modules = dict(sys.modules)
        self._sys_argv = list(sys.argv)

    def restore_sys_env(self):
        sys.path[:] = self._sys_paths
        sys.argv[:] = self._sys_argv
        sys.modules.clear()
        sys.modules.update(self._sys_modules)

    @staticmethod
    def delete(path: Union[str, Path, None] = None):
        if path is not None:
            p = Path(path)
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree('out', ignore_errors=True)
