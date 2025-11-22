import unittest
from pathlib import Path
from typing import Any

from pyfmto.experiments import Launcher
from pyfmto.utilities import save_yaml
from ..helpers import remove_temp_files
from ..helpers.generators import (
    gen_algorithm, gen_problem
)


class TestLauncher(unittest.TestCase):

    def setUp(self):
        Path('out').mkdir(exist_ok=True)
        self.conf_file = 'out/config.yaml'
        self.conf: dict[str, Any] = {
            'launcher': {
                'algorithms': ['ALG1'],
                'problems': ['PROB1'],
            }
        }
        gen_algorithm('ALG1')
        gen_problem('PROB1')
        save_yaml(self.conf, self.conf_file)

    def tearDown(self):
        remove_temp_files()

    def test_default_conf(self):
        launcher = Launcher(self.conf_file)
        launcher.run()

    def test_not_save(self):
        self.conf['launcher'].update({'save': False, 'backup': True})
        save_yaml(self.conf, self.conf_file)
        launcher = Launcher(self.conf_file)
        launcher.run()
