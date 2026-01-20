from pyfmto.problem import MultiTaskProblem
from tests.helpers.testcases import TestCaseAlgProbConf


class TestProblemData(TestCaseAlgProbConf):

    def test_default_value(self):
        prob_name = self.prob_names[0]
        prob = self.load_problem(prob_name)
        prob.params_default.pop('dim')
        self.assertIn(prob_name, prob.name)
        self.assertEqual(prob.npd, 1)
        self.assertEqual(prob.npd_str, 'NPD1')
        self.assertEqual(prob.params_diff, '')
        self.assertEqual(prob.dim, 0)
        self.assertEqual(prob.dim_str, '')
        self.assertNotIn('fe_init', prob.params)
        self.assertNotIn('fe_max', prob.params)

        prob.params_default.update({'dim': 1})
        self.assertEqual(prob.dim, 1)
        self.assertEqual(prob.dim_str, '1D')
        self.assertNotEqual(prob.params_diff, '')
        self.assertEqual(prob.params['fe_init'], 5 * prob.dim)
        self.assertEqual(prob.params['fe_max'], 11 * prob.dim)

        prob.params_update = {'dim': 2, 'npd': 2}
        self.assertEqual(prob.dim, 2)
        self.assertEqual(prob.npd, 2)
        self.assertEqual(prob.dim_str, '2D')
        self.assertEqual(prob.npd_str, 'NPD2')
        self.assertEqual(prob.params['fe_init'], 5 * prob.dim)
        self.assertEqual(prob.params['fe_max'], 11 * prob.dim)
        self.assertTrue(prob.params_diff != '')

        prob.params_update.update({'fe_init': 100, 'fe_max': 200})
        self.assertEqual(prob.params['fe_init'], 100)
        self.assertEqual(prob.params['fe_max'], 200)

        self.assertNotIn('no configurable parameters', prob.params_yaml)
        prob.params_default = {}
        self.assertIn('no configurable parameters', prob.params_yaml)
        self.assertIsInstance(prob.initialize(), MultiTaskProblem)

    def test_not_available(self):
        prob = self.load_problem('NOT_AVAILABLE')
        self.assertRaises(ValueError, prob.initialize)
        self.assertEqual(prob.task_num, 0)
