import shutil
from pathlib import Path
from unittest import TestCase

from pyfmto import list_problems

src_init = "from .my_prob import MyMTP"
src_prob = """
import numpy as np
from typing import Union
from numpy import ndarray
from pyfmto.problems import SingleTaskProblem, MultiTaskProblem


class MySTP(SingleTaskProblem):

    def __init__(self, dim=2, **kwargs):
        super().__init__(dim=dim, obj=1, lb=0, ub=1, **kwargs)

    def _eval_single(self, x: ndarray):
        return np.sum(x)


class MyMTP(MultiTaskProblem):
    is_realworld = False
    intro = "user defined MTP"
    notes = "for test purposes only"
    references = ['no reference']

    def __init__(self, dim=10, **kwargs):
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs) -> Union[list[SingleTaskProblem], tuple[SingleTaskProblem]]:
        return [MySTP(dim=dim, **kwargs) for _ in range(10)]
"""


class TestUserDefinedProblems(TestCase):
    def setUp(self):
        self.problems_dir = Path('problems')
        self.problems_dir.mkdir(exist_ok=True)
        with open(self.problems_dir / '__init__.py', 'w') as f:
            f.write(src_init)
        with open(self.problems_dir / 'my_prob.py', 'w') as f:
            f.write(src_prob)

    def tearDown(self):
        shutil.rmtree(self.problems_dir)

    def test_load_user_defined_problem(self):
        self.assertTrue('MyMTP' in list_problems())
