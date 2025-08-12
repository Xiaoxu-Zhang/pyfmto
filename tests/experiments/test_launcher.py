import shutil
import unittest
from pathlib import Path

from pyfmto import export_launcher_config, export_alg_template, export_algorithm_config, export_problem_config
from pyfmto.experiments import Launcher
from pyfmto.utilities import load_yaml, save_yaml


class TestLauncher(unittest.TestCase):

    def setUp(self):
        export_alg_template('TMP')
        export_launcher_config(algs=['TMP'], repeat=2, probs=['tetci2019'], mode='update')
        export_problem_config(probs=('tetci2019', ), mode='update')
        conf = load_yaml('config.yaml')
        conf['problems'].update(
            {
                'tetci2019': {'fe_init': 20, 'fe_max': 25, 'dim': 3}
            }
        )
        save_yaml(conf, 'config.yaml')

    def tearDown(self):
        Path('config.yaml').unlink()
        shutil.rmtree('algorithms')
        shutil.rmtree('out')

    def test_basic_run(self):
        launcher = Launcher()
        launcher.run()

    def test_kwargs_update(self):
        export_algorithm_config(algs=('TMP', ), mode='update')  # cover kwargs update logic
        conf = load_yaml('config.yaml')
        conf['algorithms']['TMP'].update({'client': {'alpha': 0.01}})
        save_yaml(conf, 'config.yaml')
        launcher = Launcher()
        launcher.run()

    def test_not_save(self):
        conf = load_yaml('config.yaml')
        conf['launcher']['save'] = False
        save_yaml(conf, 'config.yaml')
        launcher = Launcher()
        launcher.run()
