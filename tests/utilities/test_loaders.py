
from ruamel.yaml import CommentedMap

import pyfmto.experiment.config
import pyfmto.experiment.utils
import pyfmto.utilities.io
from pyfmto.experiment import list_report_formats, show_default_conf
from pyfmto.problem import ProblemData
from pyfmto.utilities import loaders
from tests.helpers import PyfmtoTestCase, gen_code


class TestUtilities(PyfmtoTestCase):

    def test_recursive_to_pure_dict(self):
        data = {
            'a': {'m': 1},
            'b': {'n': CommentedMap(e=2)},
            'c': [1, 2]
        }
        expected = {
            'a': {'m': 1},
            'b': {'n': {'e': 2}},
            'c': [1, 2]
        }
        res = pyfmto.utilities.io.recursive_to_pure_dict(data)
        self.assertIsInstance(res, dict)
        self.assertIsInstance(res['a'], dict)
        self.assertIsInstance(res['b'], dict)
        self.assertIsInstance(res['c'], list)
        self.assertIsInstance(res['b']['n'], dict)
        self.assertEqual(res, expected, msg=f"\n  result: {res}\nexpected: {expected}")

    def test_list_report_formats(self):
        res = list_report_formats(print_it=True)
        for fmt in ['curve', 'violin', 'excel', 'latex', 'console']:
            self.assertIn(fmt, res)

    def test_show_default_conf(self):
        for fmt in [*list_report_formats(), 'nonexist']:
            with self.subTest(fmt=fmt):
                self.assertIsNone(show_default_conf(fmt))

    def test_load_problem(self):
        self.save_sys_env()
        self.delete(self.tmp_dir)
        self.prob = 'PROB1'
        gen_code('problems', self.prob, self.tmp_dir)
        res = loaders.load_problem(self.prob, [str(self.tmp_dir)])
        self.assertIsInstance(res, ProblemData)
        loaders.list_problems([str(self.tmp_dir)], print_it=True)
        prob = loaders.load_problem('PROB2', [str(self.tmp_dir)])
        self.assertFalse(prob.available)
        self.delete(self.tmp_dir)
        self.restore_sys_env()
