import shutil
import sys
import unittest
from pathlib import Path
from typing import Union

from pyfmto import ConfigLoader, list_algorithms, list_problems, load_algorithm, load_problem
from pyfmto.utilities import loaders
from pyfmto.utilities.loaders import discover
from tests.helpers import gen_code
from tests.helpers.generators import gen_config


class PyfmtoTestCase(unittest.TestCase):
    def setUp(self):
        self.save_sys_env()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.auto_clear()
        self.restore_sys_env()

    @property
    def tmp_dir(self) -> Path:
        return Path('temp_dir_for_test')

    @property
    def log_dir(self) -> Path:
        return self.out_dir / 'logs'

    @property
    def out_dir(self) -> Path:
        return Path('out')

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

    def gen_algs(self, names: Union[str, list[str]]):
        gen_code('algorithms', names, self.tmp_dir)

    def gen_probs(self, names: Union[str, list[str]]):
        gen_code('problems', names, self.tmp_dir)

    def gen_config(self, content: str, name: str = 'config.yaml'):
        return gen_config(content, self.tmp_dir / name)

    @staticmethod
    def clear_cache():
        loaders._DISCOVER_CACHE.clear()

    def auto_clear(self):
        self.clear_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    @staticmethod
    def delete(path: Union[str, Path]):
        p = Path(path)
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p, ignore_errors=True)


class TestCaseAlgProbConf(PyfmtoTestCase):

    def setUp(self):
        super().setUp()
        self.alg_names = ['ALG1', 'ALG2']
        self.prob_names = ['PROB1', 'PROB2']
        self.gen_algs(self.alg_names)
        self.gen_probs(self.prob_names)
        self.conf_filename = self.gen_config(
            f"""
            launcher:
                sources: [{self.tmp_dir}]
                results: {self.tmp_dir / 'out' / 'results'}
                save: true
                repeat: 2
                algorithms: [{', '.join(self.alg_names)}]
                problems: [{self.prob_names[0]}]
            """
        )
        self.config = ConfigLoader(self.conf_filename)

    @property
    def n_alg(self):
        return len(self.alg_names)

    @property
    def n_prob(self):
        return len(self.prob_names)

    @property
    def algorithms_discovered(self):
        return discover(self.sources).get('algorithms')

    @property
    def problems_discovered(self):
        return discover(self.sources).get('problems')

    def load_algorithm(self, name: str):
        return load_algorithm(name, self.sources)

    def load_problem(self, name: str, **kwargs):
        return load_problem(name, self.sources, **kwargs)

    def list_algorithms(self):
        list_algorithms(self.sources, print_it=True)

    def list_problems(self):
        list_problems(self.sources, print_it=True)
