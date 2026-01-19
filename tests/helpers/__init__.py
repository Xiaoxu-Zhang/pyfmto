import shutil
import sys
import unittest
from pathlib import Path
from typing import Union

from pyfmto.utilities import loaders

from .generators import ExpDataGenerator, gen_code
from .launchers import running_server, start_clients

__all__ = [
    "ExpDataGenerator",
    "PyfmtoTestCase",
    "gen_code",
    "running_server",
    "start_clients",
]


class PyfmtoTestCase(unittest.TestCase):

    @property
    def tmp_dir(self) -> Path:
        return Path('temp_dir_for_test')

    @property
    def sources(self) -> list[str]:
        return [str(self.tmp_dir)]

    def save_sys_env(self):
        self._sys_paths = list(sys.path)
        self._sys_modules = dict(sys.modules)
        self._sys_argv = list(sys.argv)

    def restore_sys_env(self):
        sys.path[:] = self._sys_paths
        sys.argv[:] = self._sys_argv
        sys.modules.clear()
        sys.modules.update(self._sys_modules)

    def delete(self, path: Union[str, Path, None] = None):
        loaders._DISCOVER_CACHE.clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        if path is not None:
            p = Path(path)
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree('out', ignore_errors=True)

    @staticmethod
    def init_log_dir():
        Path('out/logs').mkdir(parents=True, exist_ok=True)
