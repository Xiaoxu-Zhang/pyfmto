import unittest
import yaml
from pathlib import Path

from pyfmto.problems import load_problem


class TestAllProblemsCanBeEvaluate(unittest.TestCase):

    def setUp(self):
        with open(Path(__file__).parents[2] / 'src' / 'pyfmto' / 'problems' / 'problems.yaml') as f:
            problems = yaml.safe_load(f)
        self.all_prob = []
        for problem_type, sub_types in problems.items():
            for sub_type, probs in sub_types.items():
                self.all_prob += list(probs.keys())

    def test_evaluate(self):
        for prob_name in self.all_prob:
            problem, _ = load_problem(prob_name)
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
