from unittest import TestCase

from pyfmto import list_problems
from tests.helpers.cleaners import remove_temp_files
from tests.helpers.generators import gen_problem


class TestUserDefinedProblems(TestCase):
    def setUp(self):
        gen_problem('PROB1')

    def tearDown(self):
        remove_temp_files()

    def test_load_user_defined_problem(self):
        self.assertTrue('PROB1' in list_problems())
