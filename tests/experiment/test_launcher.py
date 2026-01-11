from pyfmto.experiment import Launcher
from tests.experiment import ExpTestCase


class TestLauncher(ExpTestCase):

    def test_default_conf(self):
        launcher = Launcher(self.conf.launcher)
        launcher.run()
        self.assertGreater(self.conf.launcher.n_exp, 0)
        self.assertTrue(self.conf.launcher.save)
        for exp in self.conf.launcher.experiments:
            self.assertTrue(exp.root.exists())
            self.assertNotEqual(exp.root.iterdir(), [])

    def test_not_save(self):
        self.conf.config['launcher'].update({'save': False, 'verbose': True})
        self.assertFalse(self.conf.launcher.save)
        self.assertTrue(self.conf.launcher.verbose)
        launcher = Launcher(self.conf.launcher)
        launcher.run()
