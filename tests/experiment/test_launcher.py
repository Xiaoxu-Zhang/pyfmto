from pathlib import Path

from pyfmto.experiment import Launcher
from tests.experiment import ExpTestCase


class TestLauncher(ExpTestCase):

    def test_save(self):
        self.conf.config['launcher'].update({'verbose': True})
        launcher = Launcher(self.conf.launcher)
        launcher.run()
        self.assertGreater(launcher.conf.n_exp, 0)
        self.assertTrue(launcher.conf.save, msg="self.conf.launcher.save is False")
        for exp in launcher.conf.experiments:
            self.assertTrue(exp.algorithm.available)
            self.assertTrue(exp.problem.available)
            self.assertTrue(exp.success, msg=f"[{exp.algorithm.name}][{exp.problem.name}] success is False")
            self.assertTrue(exp.root.exists(), msg=f"{exp.root} not exists.")
            self.assertTrue(exp.code_dest.exists(), msg=f"{exp.code_dest} not exists.")
            self.assertTrue(exp.markdown_dest.exists(), msg=f"{exp.markdown_dest} not exists.")
            self.assertGreater(exp.num_results, 0, msg=f"[{exp.algorithm.name}][{exp.problem.name}] num_results is 0")

    def test_not_save(self):
        self.conf.config['launcher'].update({'save': False})
        launcher = Launcher(self.conf.launcher)
        self.assertFalse(launcher.conf.save, msg="self.conf.launcher.save is True")
        self.assertFalse(launcher.conf.verbose, msg="self.conf.launcher.verbose is True")
        self.assertFalse(Path(launcher.conf.results).exists(), msg="Results directory exists before launcher.run()")
        launcher.run()
        self.assertFalse(Path(launcher.conf.results).exists(), msg="Results directory exists after launcher.run()")
        for exp in launcher.conf.experiments:
            self.assertTrue(exp.success, msg=f"[{exp.algorithm.name}][{exp.problem.name}] success is False")
            self.assertFalse(exp.root.exists(), msg=f"{exp.root} exists.")
            self.assertFalse(exp.code_dest.exists(), msg=f"{exp.code_dest} exists.")
            self.assertFalse(exp.markdown_dest.exists(), msg=f"{exp.markdown_dest} exists.")
            self.assertEqual(exp.num_results, 0, msg=f"[{exp.algorithm.name}][{exp.problem.name}] num_results is not 0")
