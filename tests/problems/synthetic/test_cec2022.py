import unittest

from pyfmto import init_problem


class TestCec2022(unittest.TestCase):

    def test_valid_dim(self):
        init_problem('Cec2022', dim=10)
        init_problem('Cec2022', dim=20)

    def test_invalid_dim(self):
        self.assertRaises(ValueError, init_problem, name='Cec2022', dim=9)
