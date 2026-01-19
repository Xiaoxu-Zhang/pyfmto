from pathlib import Path

from pyfmto.experiment.config import ConfigLoader
from tests.helpers import PyfmtoTestCase, gen_code
from tests.helpers.generators import gen_config


class ExpTestCase(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        Path('out/logs').mkdir(parents=True, exist_ok=True)
        self.algs = ['ALG1', 'ALG2']
        self.probs = ['PROB1']
        gen_code('algorithms', self.algs, self.tmp_dir)
        gen_code('problems', self.probs, self.tmp_dir)
        self.conf_file = gen_config(
            f"""
            launcher:
                sources: [{self.tmp_dir}]
                results: {self.tmp_dir / 'out' / 'results'}
                save: true
                repeat: 2
                algorithms: [{', '.join(self.algs)}]
                problems: [{', '.join(self.probs)}]
            """,
            self.tmp_dir
        )
        self.conf = ConfigLoader(self.conf_file)

    def tearDown(self):
        self.delete()
        self.delete('out')
        self.restore_sys_env()
