import unittest
import pickle
from pathlib import Path
import numpy as np

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
        default, _ = load_problem('arxiv2017')
        mod_dim, _ = load_problem('arxiv2017', dim=MODIFIED_DIM)
        mod_fe, _ = load_problem('arxiv2017', init_fe=MODIFIED_INIT_FE, max_fe=MODIFIED_MAX_FE,
                                 np_per_dim=MODIFIED_NP_PER_DIM)

        self.assertEqual(len(default), 18)
        self.assertEqual(len(mod_dim), 18)
        self.assertEqual(len(mod_fe), 18)

        task_default = default[0]
        task_mod_dim = mod_dim[0]
        task_mod_fe = mod_fe[0]

        task_default.init_solutions()
        task_mod_dim.init_solutions()
        task_mod_fe.init_solutions()

        self.assertEqual(task_default.solutions.dim, DEFAULT_DIM)
        self.assertEqual(task_default.solutions.fe_init, DEFAULT_INIT_FE)
        self.assertEqual(task_default.solutions.size, DEFAULT_INIT_FE)

        self.assertEqual(task_mod_dim.solutions.dim, MODIFIED_DIM)
        self.assertEqual(task_mod_dim.solutions.fe_init, 5 * MODIFIED_DIM)
        self.assertEqual(task_mod_dim.solutions.size, 5 * MODIFIED_DIM)

        self.assertEqual(task_mod_fe.solutions.fe_init, MODIFIED_INIT_FE)

        problem10, _ = load_problem('arxiv2017', dim=10)
        problem25, _ = load_problem('arxiv2017', dim=25)
        problem30, _ = load_problem('arxiv2017', dim=30)
        problem50, _ = load_problem('arxiv2017', dim=50)
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
            {'shift_mat': ndarray|int, 'rot_mat': ndarray|int},
    }
    """

    def setUp(self):
        with open(Path(__file__).parent / "validation_arxiv2017.pkl", 'rb') as f:
            self.val_data = pickle.load(f)
        self.problems, _ = load_problem('arxiv2017')

    def test_validate_attributes(self):
        for prob in self.problems:
            name = self.val_data[prob.id]['name']
            lb = self.val_data[prob.id]['lb']
            ub = self.val_data[prob.id]['ub']
            shift_mat = self.val_data[prob.id]['trans']['shift_mat']
            rot_mat = self.val_data[prob.id]['trans']['rot_mat']
            prefix = f"\nid({prob.id})|name({name})"
            self.assertEqual(prob.name, name)
            self.assertTrue(np.all(prob.x_lb == lb), f"{prefix}\nx_lb:{prob.x_lb}\n  lb:{lb}")
            self.assertTrue(np.all(prob.x_ub == ub), f"{prefix}\nx_ub:{prob.x_ub}\n  ub:{ub}")
            if prob.shift_mat is not None:
                self.assertTrue(np.all(prob.shift_mat == shift_mat),
                                f"{prefix}\nshift_mat:{prob.shift_mat}\n  shift_mat:{shift_mat}")
            else:
                self.assertTrue(np.all(0 == shift_mat),
                                f"{prefix}\nshift_mat:{prob.shift_mat}\n  shift_mat:{shift_mat}")
            if prob.rotate_mat is not None:
                self.assertTrue(np.all(prob.rotate_mat == rot_mat),
                                f"{prefix}\nrotate_mat:{prob.rotate_mat}\n  rotate_mat:{rot_mat}")
            else:
                self.assertTrue(np.all(1 == rot_mat),
                                f"{prefix}\nrotate_mat:{prob.rotate_mat}\n  rotate_mat:{rot_mat}")

    def test_evaluation(self):
        for prob in self.problems:
            x = self.val_data[prob.id]['x']
            y = self.val_data[prob.id]['y']
            diff = np.abs(prob.evaluate(x) - y)
            self.assertTrue(np.all(diff < 1e-5), f"id({prob.id})|diff y:\n{prob.evaluate(x)-y}")