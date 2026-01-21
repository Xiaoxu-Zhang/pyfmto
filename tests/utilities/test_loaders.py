from unittest.mock import patch

from ruamel.yaml import CommentedMap

import pyfmto.experiment.config
import pyfmto.experiment.utils
import pyfmto.utilities.io
from pyfmto import list_algorithms, list_problems
from pyfmto.experiment import list_report_formats, show_default_conf
from pyfmto.problem import ProblemData
from pyfmto.utilities import loaders
from pyfmto.utilities.io import dumps_yaml, recursive_to_pure_dict
from pyfmto.utilities.loaders import discover
from tests.helpers import PyfmtoTestCase, gen_code
from tests.helpers.testcases import TestCaseAlgProbConf


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


class TestComponentUtils(TestCaseAlgProbConf):
    def test_list_components(self):
        res = list_algorithms(self.sources)
        self.assertEqual(set(self.alg_names), set(res['name']), msg=dumps_yaml(recursive_to_pure_dict(res)))
        res = list_problems(self.sources)
        self.assertEqual(set(self.prob_names), set(res['name']), msg=dumps_yaml(recursive_to_pure_dict(res)))

    def test_list_components_not_exist(self):
        res = list_algorithms([str(self.tmp_dir / 'not_exist')])
        self.assertEqual(len(res['name']), 0, msg=dumps_yaml(recursive_to_pure_dict(res)))
        res = list_problems([str(self.tmp_dir / 'not_exist')])
        self.assertEqual(len(res['name']), 0, msg=dumps_yaml(recursive_to_pure_dict(res)))


class TestDiscover(PyfmtoTestCase):
    def setUp(self):
        super().setUp()
        self.gen_algs(['ALG1', 'ALG2'])
        self.gen_probs(['PROB1', 'PROB2'])

    def test_normal_discover(self):
        res = discover(self.sources)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res['algorithms']), 2)
        self.assertEqual(len(res['problems']), 2)

    @patch('importlib.import_module')
    def test_import_raises_error(self, mock_import):
        mock_import.side_effect = ImportError
        res = discover(self.sources)
        algs = res['algorithms']
        probs = res['problems']
        self.assertEqual(len(algs), 2)
        self.assertEqual(len(probs), 2)
        self.assertFalse(algs['ALG1'][0].available)
