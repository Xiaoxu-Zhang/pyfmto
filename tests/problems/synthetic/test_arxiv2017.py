import numpy as np
import unittest
from pathlib import Path
from scipy.io import loadmat

from pyfmto import init_problem

DEFAULT_DIM = 10
DEFAULT_INIT_FE = 5 * DEFAULT_DIM
DEFAULT_MAX_FE = 11 * DEFAULT_DIM
DEFAULT_NP_PER_DIM = 1

MODIFIED_DIM = 6
MODIFIED_INIT_FE = 12
MODIFIED_MAX_FE = 24
MODIFIED_NP_PER_DIM = 4


class TestArxiv2017(unittest.TestCase):
    def test_init_default(self):
        default = init_problem('arxiv2017')
        self.assertEqual(len(default), 18)
        for task in default:
            with self.subTest(task=task):
                self.assertEqual(task.solutions.dim, DEFAULT_DIM)
                self.assertEqual(task.solutions.fe_init, DEFAULT_INIT_FE)
                self.assertEqual(task.solutions.size, DEFAULT_INIT_FE)

    def test_different_dim(self):
        for dim in [3, 5, 10, 15, 20, 30, 50]:
            with self.subTest(dim=dim):
                prob = init_problem('arxiv2017', dim=dim, _init_solutions=False)
                if dim <= 25:
                    self.assertEqual(prob.task_num, 18)
                else:
                    self.assertEqual(prob.task_num, 17)
                for task in prob:
                    self.assertEqual(task.dim, dim)
                _ = str(prob)

    def test_different_budgets(self):
        for fe_init, fe_max in [(10, 20), (30, 40), (50, 60)]:
            with self.subTest(fe_init=fe_init, fe_max=fe_max):
                prob = init_problem('arxiv2017', fe_init=fe_init, fe_max=fe_max)
                for task in prob:
                    self.assertEqual(task.fe_init, fe_init)
                    self.assertEqual(task.fe_max, fe_max)

    def test_different_np(self):
        for np_per_dim in [1, 2, 3, 4, 5]:
            with self.subTest(np_per_dim=np_per_dim):
                prob = init_problem('arxiv2017', np_per_dim=np_per_dim)
                for task in prob:
                    self.assertEqual(task.np_per_dim, np_per_dim)

    def test_rasis(self):
        dim_none = {'dim': None}
        dim_out = {'dim': 51}
        self.assertRaises(ValueError, init_problem, 'arxiv2017', **dim_none)
        self.assertRaises(ValueError, init_problem, 'arxiv2017', **dim_out)


class TestValidateFunctions(unittest.TestCase):
    """
    validation data = {
        'name': t.__class__.__name__,
        'x': np.array(res_x),
        'y': np.array(res_y),
        'lb': t.lb(int),
        'ub': t.ub(int),
        'trans':
            {'shift': ndarray|int, 'rot_mat': ndarray|int},
    }
    """

    def setUp(self):
        self.problems = init_problem('arxiv2017', _init_solutions=False)
        self.val_data = {}
        val_data = loadmat(str(Path(__file__).parent / "validation_arxiv2017.mat"))
        for p in self.problems:
            value = {
                'name': val_data[f"F{p.id}_name"],
                'x': val_data[f"F{p.id}_x"],
                'y': val_data[f"F{p.id}_y"],
                'lb': val_data[f"F{p.id}_lb"],
                'ub': val_data[f"F{p.id}_ub"],
                'trans': {
                    'rot_mat': val_data[f"F{p.id}_trans_rot"],
                    'shift': val_data[f"F{p.id}_trans_shift"]},
            }
            self.val_data[p.id] = value

    def test_evaluation(self):
        for prob in self.problems:
            x = self.val_data[prob.id]['x']
            y = self.val_data[prob.id]['y']
            diff = np.abs(prob.evaluate(x) - y)
            self.assertTrue(np.all(diff < 1e-5), f"id({prob.id})|diff y:\n{prob.evaluate(x) - y}")
