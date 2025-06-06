import unittest

from pyfmto.problems import load_problem


class TestTetci2019(unittest.TestCase):

    def test_high_dimensional(self):
        prob = load_problem('tetci2019', dim=28)
        self.assertEqual(prob.task_num, 8)
        _ = str(prob)

    def test_invalid_dim(self):
        self.assertRaises(ValueError, load_problem, name='tetci2019', dim=51)