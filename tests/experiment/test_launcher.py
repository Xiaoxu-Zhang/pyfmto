from pathlib import Path

from pyfmto.experiment import Launcher
from tests.helpers.testcases import TestCaseAlgProbConf


class TestLauncher(TestCaseAlgProbConf):

    def test_save(self):
        self.config.config['launcher'].update({'verbose': True})
        self.config.config['launcher']['algorithms'].append('INVALID')
        launcher = Launcher(self.config.launcher)
        launcher.run()
        self.assertGreater(launcher.conf.n_exp, 0)
        self.assertTrue(launcher.conf.save, msg="self.conf.launcher.save is False")
        for exp in launcher.conf.experiments:
            if not exp.available:
                continue
            self.assertTrue(exp.algorithm.available)
            self.assertTrue(exp.problem.available)
            self.assertTrue(exp.result_dir.exists(), msg=f"{exp.result_dir} not exists.")
            self.assertTrue(exp.code_dest.exists(), msg=f"{exp.code_dest} not exists.")
            self.assertTrue(exp.markdown_dest.exists(), msg=f"{exp.markdown_dest} not exists.")
            self.assertGreater(exp.n_results, 0, msg=f"[{exp.algorithm.name}][{exp.problem.name}] num_results is 0")

    def test_not_save(self):
        self.config.config['launcher'].update({'save': False})
        launcher = Launcher(self.config.launcher)
        self.assertFalse(launcher.conf.save, msg="self.conf.launcher.save is True")
        self.assertFalse(launcher.conf.verbose, msg="self.conf.launcher.verbose is True")
        self.assertFalse(Path(launcher.conf.results).exists(), msg="Results directory exists before launcher.run()")
        launcher.run()
        self.assertFalse(Path(launcher.conf.results).exists(), msg="Results directory exists after launcher.run()")
        for exp in launcher.conf.experiments:
            self.assertFalse(exp.result_dir.exists(), msg=f"{exp.result_dir} exists.")
            self.assertFalse(exp.code_dest.exists(), msg=f"{exp.code_dest} exists.")
            self.assertFalse(exp.markdown_dest.exists(), msg=f"{exp.markdown_dest} exists.")
            self.assertEqual(exp.n_results, 0, msg=f"[{exp.algorithm.name}][{exp.problem.name}] num_results is not 0")
