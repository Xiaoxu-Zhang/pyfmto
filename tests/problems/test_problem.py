import matplotlib
import numpy as np
import shutil
import unittest
import pyvista
from itertools import product
from unittest.mock import patch
from pathlib import Path

from numpy import ndarray

from tests.problems import ConstantProblem, SimpleProblem, MtpSynthetic, MtpRealworld, MtpNonIterableReturn

matplotlib.use('Agg')
pyvista.OFF_SCREEN = True


class TestProblemBase(unittest.TestCase):

    def setUp(self):
        self.dims = [2, 5, 10]
        self.tmp_dir = Path('tmp')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_attributes(self):
        for dim, obj in product(self.dims, range(2, 5)):
            prob = ConstantProblem(dim=dim, obj=obj, lb=-5, ub=5)
            self.assertTrue(prob.no_partition)
            self.assertEqual(prob.id, -1)
            self.assertFalse(prob.auto_update_solutions)
            prob.set_id(1)
            self.assertEqual(prob.id, 1)

    def test_properties(self):
        for dim, obj in product(self.dims, range(2, 5)):
            prob = ConstantProblem(dim=dim, obj=obj, lb=-5, ub=5)
            self.assertEqual(prob.dim, dim)
            self.assertEqual(prob.obj, obj)
            self.assertEqual(prob.name, "ConstantProblem")
            self.assertEqual(prob.fe_init, 5*dim)
            self.assertEqual(prob.fe_max, 11*dim)
            self.assertEqual(prob.solutions.size, 0)
            self.assertEqual(prob.fe_available, 11*dim,
                             f"fe_init={prob.fe_init}, size={prob.solutions.size}, fe_max={prob.fe_max}")
            self.assertEqual(prob.npd, 1)
            self.assertTrue(np.all(prob.lb == -5))
            self.assertTrue(np.all(prob.ub == 5))
            self.assertTrue(np.all(prob.shift == 0))
            self.assertTrue(np.all(prob.rotation == np.eye(dim)))

            _ = repr(prob)
            _ = str(prob)

    def test_init_partition(self):
        for dim, np_value in product(self.dims, [1, 2, 4, 6]):
            prob = ConstantProblem(dim=dim, obj=1, lb=-5, ub=5, **{'npd': np_value})
            self.assertTrue(prob.no_partition)
            prob.init_partition()
            self.assertFalse(prob.no_partition)
            self.assertEqual(prob._partition.shape, (2, dim))
            self.assertTrue(np.all(prob._partition[0] >= prob.lb))
            self.assertTrue(np.all(prob._partition[1] <= prob.ub))

    def test_gen_plotting_data(self):
        prob = ConstantProblem(dim=5, obj=1, lb=-1, ub=1)
        D1, D2, Z, _ = prob.gen_plot_data()
        self.assertEqual(D1.ndim, 2)
        self.assertEqual(D2.ndim, 2)
        self.assertEqual(Z.ndim, 2)
        for dim, n_points in product(self.dims, [20, 50]):
            prob = ConstantProblem(dim=dim, obj=1, lb=-5, ub=5)
            D1, D2, Z, _ = prob.gen_plot_data(n_points=n_points)
            self.assertEqual(D1.shape, (n_points, n_points))
            self.assertEqual(D2.shape, (n_points, n_points))
            self.assertEqual(Z.shape, (n_points, n_points))
            D1, D2, Z, _ = prob.gen_plot_data(scale_mode='xy')
            self.assertTrue(np.all(D1 >= 0))
            self.assertTrue(np.all(D1 <= 1))
            self.assertTrue(np.all(D2 >= 0))
            self.assertTrue(np.all(D2 <= 1))
            _ = prob.gen_plot_data(scale_mode='y')

    def test_init_solutions(self):
        for np_value in [1, 2, 4, 6]:
            prob = ConstantProblem(dim=5, obj=1, lb=-1, ub=1, **{'npd': np_value})
            self.assertTrue(prob.no_partition)
            prob.init_solutions()
            prob.init_partition()
            prob.init_solutions()  # reinitialize solutions
            self.assertEqual(prob.fe_available, prob.fe_max - prob.fe_init)
            fe_init = prob.solutions.fe_init
            init_size = prob.solutions.size
            self.assertEqual(fe_init, init_size,
                             msg=f"fe_init is {fe_init}, init_size is {init_size}")
            x = prob.solutions.x
            self.assertTrue(np.all(x >= prob.lb))
            self.assertTrue(np.all(x <= prob.ub))
            self.assertTrue(np.all(x >= prob._partition[0]), msg=f"x=\n{x}, partition=\n{prob._partition}")
            self.assertTrue(np.all(x <= prob._partition[1]))

    def test_set_x_global(self):
        prob = SimpleProblem(dim=5, obj=1, lb=-1, ub=1)
        self.assertTrue(prob.is_known_optimal)
        self.assertTrue(np.all(prob.y_global == 0))

        prob.set_x_global(None)
        self.assertFalse(prob.is_known_optimal)
        self.assertEqual(prob.x_global.size, 0)
        self.assertEqual(prob.y_global.size, 0)

        prob.set_x_global(np.arange(5))
        self.assertTrue(np.all(prob.x_global == np.arange(5)))
        self.assertTrue(np.any(prob.y_global != 0))

        with self.assertRaises(ValueError):
            prob.set_x_global([1, 2, 3, 4, 5])

        prob.set_transform(rotation=3*np.eye(5), shift=np.ones(5))
        self.assertTrue(np.all(prob.x_global == prob.inverse_transform_x(np.arange(5))))

    def test_plots(self):
        prob = ConstantProblem(dim=5, obj=1, lb=-1, ub=1)
        prob.plot_2d(n_points=10)
        prob.plot_3d(n_points=10)
        prob.plot_2d(n_points=10, filename=self.tmp_dir / 'tmp2d.png')
        prob.plot_3d(n_points=10, filename=self.tmp_dir / 'tmp3d.png')
        self.assertTrue((self.tmp_dir / 'tmp2d.png').exists())
        self.assertTrue((self.tmp_dir / 'tmp3d.png').exists())
        prob.iplot_3d(n_points=10)
        plotter = pyvista.Plotter()
        prob.iplot_3d(n_points=10, plotter=plotter, color='red')

        with patch.dict('sys.modules', {'pyvista': None}):
            prob.iplot_3d(n_points=10)

    def test_norm_denorm_methods(self):
        prob = ConstantProblem(dim=2, obj=1, lb=-1, ub=1)
        x_in_src_bound = prob.random_uniform_x(size=50)
        x_out_src_bound = x_in_src_bound - 2
        x_out_normal_bound = x_in_src_bound

        x_in_src_bound_normalized: ndarray = prob.normalize_x(x_in_src_bound)
        x_in_src_bound_denormalized = prob.denormalize_x(x_in_src_bound_normalized)
        x_out_src_bound_normalized: ndarray = prob.normalize_x(x_out_src_bound)
        x_out_normal_bound_denormalized = prob.denormalize_x(x_out_normal_bound)

        self.assertTrue(np.all(x_in_src_bound_normalized >= 0))
        self.assertTrue(np.all(x_in_src_bound_normalized <= 1))
        self.assertTrue(np.all(x_out_src_bound_normalized >= 0))
        self.assertTrue(np.all(x_out_src_bound_normalized <= 1))

        self.assertTrue(np.all(x_in_src_bound_denormalized >= prob.lb))
        self.assertTrue(np.all(x_in_src_bound_denormalized <= prob.ub))
        self.assertTrue(np.all(x_out_normal_bound_denormalized >= prob.lb))
        self.assertTrue(np.all(x_out_normal_bound_denormalized <= prob.ub))
        err: ndarray = x_in_src_bound - x_in_src_bound_denormalized
        self.assertTrue(np.all(err < 1e-10), msg=f"{err < 1e-10}")

    def test_y_norm_denorm_methods(self):
        prob = SimpleProblem(dim=5, obj=1, lb=-1, ub=1)
        prob.init_solutions()
        y_test = np.random.uniform(
            low=prob.solutions.y_min-abs(prob.solutions.y_min),
            high=prob.solutions.y_max+abs(prob.solutions.y_max),
            size=(100, 1)
        )
        self.assertTrue(np.any(y_test < prob.solutions.y_min))
        self.assertTrue(np.any(y_test > prob.solutions.y_max))
        y_norm = prob.normalize_y(y_test)
        self.assertTrue(np.all(y_norm <= 1.0))
        self.assertTrue(np.all(y_norm >= 0.0))
        y_norm = np.random.uniform(low=0.0, high=1.0, size=(100, 1))
        y_denorm = prob.denormalize_y(y_norm)
        self.assertTrue(np.all(y_denorm >= prob.solutions.y_min))
        self.assertTrue(np.all(y_denorm <= prob.solutions.y_max))

    def test_clip(self):
        prob = ConstantProblem(dim=5, obj=1, lb=-1, ub=1)
        x = [-1.5, 0., .5, 1.2, 1.1]
        self.assertTrue(np.all(prob.clip_x(x) == np.array([-1., 0., .5, 1., 1.])))

    def test_uniform_solution(self):
        n_points = 50
        # We don't test np_value=1 which is equal to no partition,
        # it will cause failure when we test np.any(x < partition[0]).
        for np_value in [2, 3, 4, 5]:
            prob = ConstantProblem(dim=5, obj=1, lb=-1, ub=1, **{'npd': np_value})
            prob.init_partition()
            x: ndarray = prob.random_uniform_x(size=n_points)
            self.assertEqual(x.shape[0], n_points)
            self.assertTrue(np.all(x >= prob._partition[0]))
            self.assertTrue(np.all(x <= prob._partition[1]))
            x = prob.random_uniform_x(size=n_points, within_partition=False)
            out_p_lb = np.any(x < prob._partition[0])
            out_p_ub = np.any(x > prob._partition[1])
            self.assertTrue(out_p_lb or out_p_ub)

    def test_auto_update_solutions(self):
        dim = 5
        prob = ConstantProblem(dim=dim, obj=1, lb=-5, ub=5)

        prob.init_solutions()
        self.assertEqual(prob.solutions.size, prob.fe_init)
        x = prob.random_uniform_x(1)
        y = prob.evaluate(x)
        self.assertEqual(prob.solutions.size, prob.fe_init)
        prob.solutions.append(x, y)
        self.assertEqual(prob.solutions.size, prob.fe_init + 1)

        prob.init_solutions()
        x = prob.random_uniform_x(10)
        y = prob.evaluate(x)
        prob.solutions.append(x, y)
        self.assertEqual(prob.solutions.size, prob.fe_init + 10)

        prob.auto_update_solutions = True

        prob.init_solutions()
        self.assertEqual(prob.solutions.size, prob.fe_init)
        x = prob.random_uniform_x(1)
        prob.evaluate(x)
        self.assertEqual(prob.solutions.size, prob.fe_init + 1)

        prob.init_solutions()
        x = prob.random_uniform_x(10)
        prob.evaluate(x)
        self.assertEqual(prob.solutions.size, prob.fe_init+10)


class TestMultiTaskProblem(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        self.tmp_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init(self):
        self.assertRaises(TypeError, MtpNonIterableReturn)
        _ = MtpSynthetic(random_ctrl='no')
        _ = MtpRealworld(random_ctrl='strong')
        self.assertRaises(ValueError, MtpRealworld, random_ctrl='not_support')
        prob = MtpSynthetic()
        filename = self.tmp_dir / 'test_show.png'
        prob.plot_distribution(filename=str(filename))
        prob.plot_distribution()
        self.assertTrue(filename.exists())

    def test_plots(self):
        mtp = MtpSynthetic()
        mtp.plot_distribution()
        mtp.plot_similarity_heatmap()
        mtp.plot_similarity_heatmap(triu='lower')
        mtp.plot_similarity_heatmap(triu='upper')
        mtp.plot_similarity_heatmap(filename=self.tmp_dir / 'test_show.png')
        self.assertTrue((self.tmp_dir / 'test_show.png').exists())
        mtp.iplot_tasks_3d(tasks_id=(1, 2), shape=(1, 2))
        with self.assertRaises(ValueError):
            mtp.plot_similarity_heatmap(method='not_support')
        with self.assertRaises(ValueError):
            mtp.iplot_tasks_3d(tasks_id=(5, 6), shape=(1, 2))
        with self.assertRaises(ValueError):
            mtp.iplot_tasks_3d(tasks_id=(1, 2), shape=(1, 1))
        with patch.dict('sys.modules', {'pyvista': None}):
            mtp.iplot_tasks_3d(tasks_id=(1, 2), shape=(1, 2))

    def test_attributes(self):
        realworld = MtpRealworld()
        synthetic = MtpSynthetic()
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
