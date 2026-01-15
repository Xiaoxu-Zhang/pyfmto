from pathlib import Path

from pyfmto.utilities.loaders import ConfigLoader
from tests.helpers import PyfmtoTestCase
from tests.helpers.generators import gen_code, gen_config


class LoadersTestCase(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        self.tmp_dir = Path('temp_dir_for_test')
        self.conf_file = gen_config(
            f"""
            launcher:
                sources: [{self.tmp_dir}]
            """,
            self.tmp_dir
        )
        self.algs = ['ALG1']
        self.probs = ['PROB1']
        gen_code('algorithms', self.algs, self.tmp_dir)
        gen_code('problems', self.probs, self.tmp_dir)
        self.conf = ConfigLoader(self.conf_file)

    def tearDown(self):
        self.delete(self.tmp_dir)
        self.restore_sys_env()

    @property
    def algorithms(self):
        return self.conf.algorithms

    @property
    def problems(self):
        return self.conf.problems
