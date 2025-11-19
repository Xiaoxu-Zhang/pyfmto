import unittest

from pyfmto import init_problem, list_problems


class TestAllProblems(unittest.TestCase):
    def setUp(self):
        self.problems = []
        for prob_name in list_problems():
            self.problems.append(init_problem(prob_name, _init_solutions=False))

    def test_list_problems(self):
        list_problems(print_it=True)

    def test_property_is_set(self):
        for prob in self.problems:
            msg = f"Problem {prob.name}'s is_realworld property not set."
            self.assertTrue(prob.is_realworld in [True, False], msg=msg)

    def test_docstring(self):
        for prob in self.problems:
            _ = str(prob)

    def test_evaluate(self):
        for prob in self.problems:
            for func in prob:
                msg = f"Problem {prob.name}.func{func.id}({func.name}) evaluation failed."
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
        self.assertRaises(ValueError, init_problem, 'invalid_name')
