import matplotlib
import numpy as np
import shutil
import unittest
from pathlib import Path

from pyfmto.problems.problem import (
    SingleTaskProblem as _SingleTaskProblem,
    MultiTaskProblem as _MultiTaskProblem
)


matplotlib.use('Agg')
TASK_NUM = 4

class STP(_SingleTaskProblem):

    def __init__(self, dim: int, obj: int, x_lb, x_ub, **kwargs):
        super().__init__(dim, obj, x_lb, x_ub, **kwargs)

    def _eval_single(self, x):
        return np.sin(np.sum(x ** 2))


class TestSingleTaskProblem(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init_instance(self):
        self.assertRaises(ValueError, STP, dim=1., obj=1, x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=1, obj=1., x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=0, obj=1, x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=-1, obj=1, x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=1, obj=0, x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=1, obj=-1, x_lb=-1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=1, obj=1, x_lb=1, x_ub=1)
        self.assertRaises(ValueError, STP, dim=1, obj=1, x_lb=[0, 0], x_ub=1)
        self.assertRaises(ValueError, STP, dim=2, obj=1, x_lb=[0, 0], x_ub=[1])
        self.assertRaises(ValueError, STP, dim=2, obj=1, x_lb=[0, 0], x_ub=[[1, 1]])
        self.assertRaises(ValueError, STP, dim=2, obj=1, x_lb=-1, x_ub=1, unexpected_arg=1)
        stp = STP(dim=2, obj=1, x_lb=-1, x_ub=1)
        self.assertRaises(TypeError, stp.set_transform, rot_mat=[0, 0], shift_mat=None)
        self.assertRaises(TypeError, stp.set_transform, rot_mat=None, shift_mat='0')

    def test_attributes(self):
        stp = STP(dim=2, obj=1, x_lb=-1, x_ub=1)
        self.assertTrue(stp.name in stp.__repr__())
        self.assertTrue(stp.name in stp.__str__())

    def test_norm_denorm_methods(self):
        stp = STP(dim=2, obj=1, x_lb=-1, x_ub=1)
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

        self.assertTrue(np.all(x_in_src_bound_denormalized >= stp.x_lb))
        self.assertTrue(np.all(x_in_src_bound_denormalized <= stp.x_ub))
        self.assertTrue(np.all(x_out_normal_bound_denormalized >= stp.x_lb))
        self.assertTrue(np.all(x_out_normal_bound_denormalized <= stp.x_ub))
        err = x_in_src_bound - x_in_src_bound_denormalized
        self.assertTrue(np.all(err < 1e-10), msg=f"{err < 1e-10}")

    def test_uniform_solution(self):

        stp_np1 = STP(dim=5, obj=1, x_lb=-1, x_ub=1, **{'np_per_dim': 1})
        stp_np2 = STP(dim=5, obj=1, x_lb=-1, x_ub=1, **{'np_per_dim': 2})
        stp_np1.init_partition()
        stp_np2.init_partition()
        x1 = stp_np1.random_uniform_x(size=100)
        x2 = stp_np2.random_uniform_x(size=100)
        self.assertEqual(x1.shape[0], 100)
        self.assertEqual(x2.shape[0], 100)

    def test_init_solutions(self):
        stp = STP(dim=5, obj=1, x_lb=-1, x_ub=1)
        stp.init_solutions()
        self.assertEqual(stp.fe_available, stp.fe_max - stp.fe_init)
        self.assertTrue(stp.no_partition)

        stp.init_solutions()  # reinitialize solutions
        init_fe = stp.solutions.fe_init
        init_size = stp.solutions.size
        self.assertEqual(stp.solutions.fe_init, stp.solutions.size,
                         msg=f"init_fe is {init_fe}, init_size is {init_size}")
        self.assertTrue(np.all(stp.solutions.x >= stp.x_lb))
        self.assertTrue(np.all(stp.solutions.x <= stp.x_ub))

        for np_per_dim in range(2, 10):
            stp_pb = STP(dim=5, obj=1, x_lb=[-1, -2, -3, -4, -5], x_ub=[6, 7, 8, 9, 10], **{'np_per_dim': np_per_dim})
            stp_pb.init_partition()
            self.assertEqual(stp_pb.np_per_dim, np_per_dim)
            band_partition = stp_pb._partition[1] - stp_pb._partition[0]
            band_bounds = stp_pb.x_ub - stp_pb.x_lb
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
        stp = STP(dim=1, obj=1, x_lb=-1, x_ub=1)
        stp.evaluate(np.array([1]))
        self.assertRaises(TypeError, stp.evaluate, '1')
        self.assertRaises(ValueError, stp.evaluate, np.array([[[1]]]))
        self.assertRaises(ValueError, stp.evaluate, np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, stp.evaluate, np.array([1,2]))

    def test_visualize(self):
        stp1 = STP(dim=1, obj=1, x_lb=-1, x_ub=1)
        stp2 = STP(dim=2, obj=1, x_lb=-1, x_ub=1)
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
        prob.plot_distribution(str(filename))
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
