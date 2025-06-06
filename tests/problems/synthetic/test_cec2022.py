import unittest

from pyfmto.problems import load_problem


class TestCec2022(unittest.TestCase):

    def test_valid_dim(self):
        load_problem('cec2022', dim=10)
        load_problem('cec2022', dim=20)

    def test_invalid_dim(self):
        self.assertRaises(ValueError, load_problem, name='cec2022', dim=9)
