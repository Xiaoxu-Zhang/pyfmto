import unittest

from pyfmto import init_problem


class TestTetci2019(unittest.TestCase):

    def test_high_dimensional(self):
        prob = init_problem('Tetci2019', dim=28)
        self.assertEqual(prob.task_num, 8)
        _ = str(prob)

    def test_invalid_dim(self):
        self.assertRaises(ValueError, init_problem, name='Tetci2019', dim=51)
