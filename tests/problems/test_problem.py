import matplotlib
import numpy as np
import shutil
import unittest
from itertools import product
from pathlib import Path

from pyfmto.problems.problem import (
    SingleTaskProblem as _SingleTaskProblem,
    MultiTaskProblem as _MultiTaskProblem,
    STPConfig, TransformerConfig
)

matplotlib.use('Agg')
TASK_NUM = 4

class TestSTPConfig(unittest.TestCase):

    def test_default(self):
        constants = {"obj": 1, "lb": -1, "ub": 1}
        test_cases = [
            ({"dim": d}, {'fe_init': 5*d, 'fe_max': 11*d})
            for d in [1, 2, 3, 4, 5]
        ]
        for case, expected in test_cases:
            stp_config = STPConfig(**case, **constants)
            self.assertEqual(stp_config.fe_init, expected['fe_init'])
            self.assertEqual(stp_config.fe_max, expected['fe_max'])
            self.assertEqual(stp_config.np_per_dim, 1)

    def test_valid_default(self):
        fe_init = range(1, 10)
        fe_max = [2*x for x in fe_init]
        valid_np = range(1, 10)
        test_cases = [
                {'fe_init': a, 'fe_max': b, 'np_per_dim': c}
                for a, b, c in zip(fe_init, fe_max, valid_np)
        ]
        for case in test_cases:
            stp_config = STPConfig(**case, **{"dim": 2, "obj": 1, "lb": -1, "ub": 1})
            self.assertEqual(stp_config.fe_init, case['fe_init'])
            self.assertEqual(stp_config.fe_max, case['fe_max'])
            self.assertEqual(stp_config.np_per_dim, case['np_per_dim'])

    def test_valid_required_inputs(self):
        valid_dims = range(1, 5)
        valid_objs = range(1, 5)
        valid_bounds = list(zip([-1, -2], [1, 2]))
        for dim, obj, (lb, ub) in product(valid_dims, valid_objs, valid_bounds):
            stp_config = STPConfig(dim=dim, obj=obj, lb=lb, ub=ub)
            self.assertEqual(stp_config.dim, dim)
            self.assertEqual(stp_config.obj, obj)
            self.assertTrue(np.all(stp_config.lb==lb*np.ones(dim)))
            self.assertTrue(np.all(stp_config.ub==ub*np.ones(dim)))

    def test_invalid_required_inputs(self):
        invalid_dims = [0.1, 1.1, -1, -2]
        for dim in invalid_dims:
            self.assertRaises(ValueError, STPConfig, **{'dim': dim, 'obj': 1, 'lb': -1, 'ub': 1})

        invalid_objs = [0.1, 1.1, -1, -2]
        for obj in invalid_objs:
            self.assertRaises(ValueError, STPConfig, **{'dim': 1, 'obj': obj, 'lb': -1, 'ub': 1})

        invalid_bounds = [
            (1, 1), (2, -2), ([-2, 2], [2, -2]), # invalid value
            ([1], [1]), ([-1], [1, 1]), ([-2, -2], [2]), ([-1, -1, -1], [1, 1, 1]) # invalid shape for dim=2
        ]
        for lb, ub in invalid_bounds:
            self.assertRaises(ValueError, STPConfig, **{'dim': 2, 'obj': 1, 'lb': lb, 'ub': ub})

    def test_init_invalid_optionals(self):
        invalid_budgets = [
            (0, 10), (10, 0), (-10, 10), (-20, -10), # fe_init or fe_max is non-positive
            (10, 9) # fe_init > fe_max
        ]
        for fe_init, fe_max in invalid_budgets:
            self.assertRaises(ValueError, STPConfig, **{'dim': 2, 'obj': 1, 'lb': 0, 'ub': 1, 'fe_init': fe_init, 'fe_max': fe_max})

        invalid_np = [-1, 0]
        for np_per_dim in invalid_np:
            self.assertRaises(ValueError, STPConfig, **{'dim': 2, 'obj': 1, 'lb': 0, 'ub': 1, 'np_per_dim': np_per_dim})


class TestTransformer(unittest.TestCase):

    def test_default(self):
        trans = TransformerConfig(dim=10)
        self.assertEqual(trans.dim, 10)
        self.assertTrue(np.all(trans.rotation == np.eye(10)))
        self.assertTrue(np.all(trans.rotation_inv == np.eye(10)))
        self.assertTrue(np.all(trans.shift == np.zeros(10)))

    def test_valid(self):
        dims = [2,3,4,5]
        scales = [0.5, 1.0, 2.0, 5.0]
        test_cases = [
            ({'dim': dim, 'rotation': np.eye(dim)*scale, 'shift': np.ones(dim)*scale}) for dim, scale in zip(dims, scales)
        ]
        for case, dim, scale in zip(test_cases, dims, scales):
            trans = TransformerConfig(**case)
            self.assertTrue(np.all(trans.rotation == np.eye(dim)*scale))
            self.assertTrue(np.all(trans.shift == np.ones(dim)*scale))
            self.assertTrue(np.all(trans.rotation_inv @ trans.rotation == np.eye(dim)))

    def test_invalid(self):
        rot_dim_mismatch = {'dim': 2, 'rotation': np.eye(3)}
        shift_dim_mismatch = {'dim': 2, 'shift': np.ones(3)}

        self.assertRaises(ValueError, TransformerConfig, **rot_dim_mismatch)
        self.assertRaises(ValueError, TransformerConfig, **shift_dim_mismatch)



class STP(_SingleTaskProblem):

    def __init__(self, dim: int, obj: int, lb, ub, **kwargs):
        super().__init__(dim, obj, lb, ub, **kwargs)

    def _eval_single(self, x):
        return np.sin(np.sum(x ** 2))


class TestSingleTaskProblem(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_attributes(self):
        stp = STP(dim=2, obj=1, lb=-1, ub=1)
        self.assertTrue(stp.name in stp.__repr__())
        self.assertTrue(stp.name in stp.__str__())

    def test_norm_denorm_methods(self):
        stp = STP(dim=2, obj=1, lb=-1, ub=1)
        x_in_src_bound = stp.random_uniform_x(size=50)
        x_out_src_bound = x_in_src_bound - 2
        x_out_normal_bound = x_in_src_bound

        x_in_src_bound_normalized = stp.normalize_x(x_in_src_bound)
        x_in_src_bound_denormalized = stp.denormalize_x(x_in_src_bound_normalized)
        x_out_src_bound_normalized = stp.normalize_x(x_out_src_bound)
        x_out_normal_bound_denormalized = stp.denormalize_x(x_out_normal_bound)

        self.assertTrue(np.all(x_in_src_bound_normalized >= 0))
        self.assertTrue(np.all(x_in_src_bound_normalized <= 1))
        self.assertTrue(np.all(x_out_src_bound_normalized >= 0))
        self.assertTrue(np.all(x_out_src_bound_normalized <= 1))

        self.assertTrue(np.all(x_in_src_bound_denormalized >= stp.lb))
        self.assertTrue(np.all(x_in_src_bound_denormalized <= stp.ub))
        self.assertTrue(np.all(x_out_normal_bound_denormalized >= stp.lb))
        self.assertTrue(np.all(x_out_normal_bound_denormalized <= stp.ub))
        err = x_in_src_bound - x_in_src_bound_denormalized
        self.assertTrue(np.all(err < 1e-10), msg=f"{err < 1e-10}")

    def test_uniform_solution(self):
        stp_np1 = STP(dim=5, obj=1, lb=-1, ub=1, **{'np_per_dim': 1})
        stp_np2 = STP(dim=5, obj=1, lb=-1, ub=1, **{'np_per_dim': 2})
        stp_np1.init_partition()
        stp_np2.init_partition()
        x1 = stp_np1.random_uniform_x(size=100)
        x2 = stp_np2.random_uniform_x(size=100)
        self.assertEqual(x1.shape[0], 100)
        self.assertEqual(x2.shape[0], 100)

    def test_init_solutions(self):
        stp = STP(dim=5, obj=1, lb=-1, ub=1)
        stp.init_solutions()
        self.assertEqual(stp.fe_available, stp.fe_max - stp.fe_init)
        self.assertTrue(stp.no_partition)

        stp.init_solutions()  # reinitialize solutions
        init_fe = stp.solutions.fe_init
        init_size = stp.solutions.size
        self.assertEqual(stp.solutions.fe_init, stp.solutions.size,
                         msg=f"init_fe is {init_fe}, init_size is {init_size}")
        self.assertTrue(np.all(stp.solutions.x >= stp.lb))
        self.assertTrue(np.all(stp.solutions.x <= stp.ub))

        for np_per_dim in range(2, 10):
            stp_pb = STP(dim=5, obj=1, lb=[-1, -2, -3, -4, -5], ub=[6, 7, 8, 9, 10], **{'np_per_dim': np_per_dim})
            stp_pb.init_partition()
            self.assertEqual(stp_pb.np_per_dim, np_per_dim)
            band_partition = stp_pb._partition[1] - stp_pb._partition[0]
            band_bounds = stp_pb.ub - stp_pb.lb
            band_expected = band_bounds / np_per_dim
            band_diff = np.abs(band_partition - band_expected)
            msg = (f"partition lb: {stp_pb._partition[0]}\n"
                   f"partition ub: {stp_pb._partition[1]}\n"
                   f"band partition: {band_partition}\n"
                   f"band bounds: {band_bounds}\n"
                   f"band expected: {band_expected}\n"
                   f"band diff: {band_diff}")
            self.assertTrue(np.all(band_diff < 1e-10), msg)

            stp_pb.init_solutions()
            self.assertTrue(np.all(stp_pb.solutions.x >= stp_pb._partition[0]))
            self.assertTrue(np.all(stp_pb.solutions.x <= stp_pb._partition[1]))

    def test_evaluation(self):
        stp = STP(dim=1, obj=1, lb=-1, ub=1)
        stp.evaluate(np.array([1]))
        self.assertRaises(TypeError, stp.evaluate, '1')
        self.assertRaises(ValueError, stp.evaluate, np.array([[[1]]]))
        self.assertRaises(ValueError, stp.evaluate, np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, stp.evaluate, np.array([1,2]))

    def test_visualize(self):
        stp1 = STP(dim=1, obj=1, lb=-1, ub=1)
        stp2 = STP(dim=2, obj=1, lb=-1, ub=1)
        vis_2d = self.tmp_dir / 'test_vis_2d'
        vis_3d = self.tmp_dir / 'test_vis_3d'
        stp2.plot_2d(n_points=10)
        stp2.plot_3d(n_points=10)
        stp2.plot_2d(filename=str(vis_2d), n_points=10)
        stp2.plot_3d(filename=str(vis_3d), n_points=10)
        self.assertTrue(vis_2d.with_suffix('.png').exists(), msg="Visualization 2d failed")
        self.assertTrue(vis_3d.with_suffix('.png').exists(), msg="Visualization 3d failed")
        self.assertRaises(ValueError, stp1.plot_2d)


class InitAttrAfterSuper(_MultiTaskProblem):
    is_realworld = False

    def __init__(self, dim: int = 2, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.test_attr = 'test_attr'

    def _init_tasks(self, dim, *args, **kwargs):
        self.test_attr = self.test_attr + 'new_value'
        return [STP(dim, 1, 0, 1, **kwargs) for _ in range(TASK_NUM)]


class InitWithInvalidReturn(_MultiTaskProblem):
    is_realworld = False

    def __init__(self):
        super().__init__()

    def _init_tasks(self, *args, **kwargs):
        return None


class SyntheticMtp(_MultiTaskProblem):
    is_realworld = False

    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(dim, **kwargs)

    def _init_tasks(self, dim, **kwargs):
        return [STP(dim, 1, 0, 1, **kwargs) for _ in range(TASK_NUM)]


class RealworldMtp(_MultiTaskProblem):
    is_realworld = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_tasks(self, **kwargs):
        return [STP(2, 1, 0, 1, **kwargs) for _ in range(TASK_NUM)]


class TestMultiTaskProblem(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        self.tmp_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init(self):
        self.assertRaises(AttributeError, InitAttrAfterSuper)
        self.assertRaises(TypeError, InitWithInvalidReturn)
        _ = SyntheticMtp(random_ctrl='no')
        _ = RealworldMtp(random_ctrl='strong')
        self.assertRaises(ValueError, RealworldMtp, random_ctrl='not_support')
        prob = SyntheticMtp()
        filename = self.tmp_dir / 'test_show.png'
        prob.plot_distribution(filename=str(filename))
        prob.plot_distribution()
        self.assertTrue(filename.exists())

    def test_attributes(self):
        realworld = RealworldMtp()
        synthetic = SyntheticMtp()
        self.assertEqual(realworld.task_num, synthetic.task_num)
        realworld.__iter__()
        synthetic.__iter__()
        _ = str(realworld), synthetic.__repr__()
        _ = str(realworld), realworld.__repr__()
        _ = realworld[0], synthetic[0]
        _ = realworld[:2], synthetic[:2]

        self.assertEqual(realworld.is_realworld, True)
        self.assertEqual(synthetic.is_realworld, False)

        for idx in range(len(realworld)):
            self.assertEqual(realworld[idx].id, idx + 1)
            self.assertEqual(synthetic[idx].id, idx + 1)
        self.assertRaises(IndexError, realworld.__getitem__, -1)
        self.assertRaises(IndexError, realworld.__getitem__, len(realworld))
        self.assertRaises(TypeError, realworld.__getitem__, '1')