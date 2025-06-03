import unittest
import yaml
from pathlib import Path

from pyfmto.problems import load_problem, list_problems


class TestAllProblemsCanBeEvaluate(unittest.TestCase):

    def test_property_is_set(self):
        for prob_name in list_problems(print_it=False):
            problem = load_problem(prob_name, _init_solutions=False)
            msg = f"Problem {problem.name}'s is_realworld property not set."
            self.assertTrue(problem.is_realworld in [True, False], msg=msg)

    def test_evaluate(self):
        for prob_name in list_problems(print_it=False):
            problem = load_problem(prob_name, _init_solutions=False)
            for func in problem:
                msg = f"Problem {prob_name}.func{func.id}({func.name}) evaluation failed."
                x1 = func.random_uniform_x(1)
                x2 = func.random_uniform_x(2)
                y1 = func.evaluate(x1)
                y2 = func.evaluate(x2)
                self.assertEqual(x1.ndim, 2, msg)
                self.assertEqual(x2.ndim, 2, msg)
                self.assertEqual(y1.ndim, 2, msg)
                self.assertEqual(y2.ndim, 2, msg)
                self.assertEqual(x1.shape[0], 1, msg)
                self.assertEqual(x2.shape[0], 2, msg)
                self.assertEqual(x1.shape[0], y1.shape[0], msg)
                self.assertEqual(x2.shape[0], y2.shape[0], msg)
        self.assertRaises(ValueError, load_problem, 'invalid_name')
