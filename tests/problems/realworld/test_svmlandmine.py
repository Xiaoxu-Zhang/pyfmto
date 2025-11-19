import unittest

from pyfmto import init_problem


class TestSvmLandmine(unittest.TestCase):
    def test_init_svm_landmine(self):
        problem = init_problem('SvmLandmine', _init_solutions=False)
        self.assertEqual(29, len(problem))
        task = problem[0]
        x_test = task.random_uniform_x(10)
        y_test = task.evaluate(x_test)
        _ = task.evaluate(x_test[0])
        self.assertEqual(10, y_test.shape[0])
        self.assertEqual(1, y_test.shape[1])
        _ = str(problem)
