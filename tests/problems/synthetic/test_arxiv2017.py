import numpy as np
import unittest
from pathlib import Path
from scipy.io import loadmat

from pyfmto.problems import load_problem


DEFAULT_DIM = 10
DEFAULT_INIT_FE = 5 * DEFAULT_DIM
DEFAULT_MAX_FE = 11 * DEFAULT_DIM
DEFAULT_NP_PER_DIM = 1

MODIFIED_DIM = 6
MODIFIED_INIT_FE = 12
MODIFIED_MAX_FE = 24
MODIFIED_NP_PER_DIM = 4


class TestArxiv2017(unittest.TestCase):
    def test_init(self):
        default = load_problem('arxiv2017')
        mod_dim = load_problem('arxiv2017', dim=MODIFIED_DIM)
        mod_fe = load_problem('arxiv2017', fe_init=MODIFIED_INIT_FE, fe_max=MODIFIED_MAX_FE,
                                 np_per_dim=MODIFIED_NP_PER_DIM)

        self.assertEqual(len(default), 18)
        self.assertEqual(len(mod_dim), 18)
        self.assertEqual(len(mod_fe), 18)

        task_default = default[0]
        task_mod_dim = mod_dim[0]
        task_mod_fe = mod_fe[0]

        self.assertEqual(task_default.solutions.dim, DEFAULT_DIM)
        self.assertEqual(task_default.solutions.fe_init, DEFAULT_INIT_FE)
        self.assertEqual(task_default.solutions.size, DEFAULT_INIT_FE)

        self.assertEqual(task_mod_dim.solutions.dim, MODIFIED_DIM)
        self.assertEqual(task_mod_dim.solutions.fe_init, 5 * MODIFIED_DIM)
        self.assertEqual(task_mod_dim.solutions.size, 5 * MODIFIED_DIM)

        self.assertEqual(task_mod_fe.solutions.fe_init, MODIFIED_INIT_FE)

        problem10 = load_problem('arxiv2017', dim=10, _init_solutions=False)
        problem25 = load_problem('arxiv2017', dim=25, _init_solutions=False)
        problem30 = load_problem('arxiv2017', dim=30, _init_solutions=False)
        problem50 = load_problem('arxiv2017', dim=50, _init_solutions=False)
        _ = str(problem10)

        self.assertEqual(problem10[0].dim, 10)
        self.assertEqual(problem25[0].dim, 25)
        self.assertEqual(problem30[0].dim, 30)
        self.assertEqual(problem50[0].dim, 50)
        self.assertEqual(len(problem10), 18)
        self.assertEqual(len(problem25), 18)
        self.assertEqual(len(problem30), 17)
        self.assertEqual(len(problem50), 17)

    def test_rasis(self):
        dim_none = {'dim': None}
        dim_out = {'dim': 51}
        self.assertRaises(ValueError, load_problem, 'arxiv2017', **dim_none)
        self.assertRaises(ValueError, load_problem, 'arxiv2017', **dim_out)


class TestValidateFunctions(unittest.TestCase):
    """
    validation data = {
        'name': t.__class__.__name__,
        'x': np.array(res_x),
        'y': np.array(res_y),
        'lb': t.x_lb(int),
        'ub': t.x_ub(int),
        'trans':
            {'shift': ndarray|int, 'rot_mat': ndarray|int},
    }
    """

    def setUp(self):
        self.problems = load_problem('arxiv2017', _init_solutions=False)
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