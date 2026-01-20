from pyfmto import load_problem
from pyfmto.experiment.config import ConfigLoader
from pyfmto.utilities.loaders import discover, list_algorithms, list_problems, load_algorithm
from tests.helpers import PyfmtoTestCase
from tests.helpers.generators import gen_code, gen_config


class LoadersTestCase(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
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
    def sources(self) -> list[str]:
        return self.conf.sources

    @property
    def algorithms(self):
        return discover(self.sources).get('algorithms')

    @property
    def problems(self):
        return discover(self.sources).get('problems')

    def load_algorithm(self, name: str):
        return load_algorithm(name, self.sources)

    def load_problem(self, name: str):
        return load_problem(name, self.sources)

    def list_algorithms(self):
        list_algorithms(self.sources, print_it=True)

    def list_problems(self):
        list_problems(self.sources, print_it=True)
