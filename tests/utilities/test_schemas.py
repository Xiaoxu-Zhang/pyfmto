import unittest
from itertools import product

import numpy as np

from pyfmto.utilities.schemas import (
    FunctionInputs,
    PlottingArgs,
    STPConfig,
    TransformerConfig,
)


class TestSTPConfig(unittest.TestCase):

    def test_default(self):
        for d in range(1, 11):
            stp_config = STPConfig(dim=d, obj=1, lb=-1, ub=1)
            self.assertEqual(stp_config.fe_init, 5 * d)
            self.assertEqual(stp_config.fe_max, 11 * d)
            self.assertEqual(stp_config.npd, 1)

    def test_valid_default(self):
        valid_np = range(1, 10)
        fe_init = range(1, 10)
        fe_max = [2 * x for x in fe_init]
        for a, b, c in zip(fe_init, fe_max, valid_np):
            stp_config = STPConfig(fe_init=a, fe_max=b, npd=c, dim=2, obj=1, lb=-1, ub=1)
            self.assertEqual(stp_config.fe_init, a)
            self.assertEqual(stp_config.fe_max, b)
            self.assertEqual(stp_config.npd, c)

    def test_valid_required_inputs(self):
        valid_dims = range(1, 5)
        valid_objs = range(1, 5)
        valid_bounds = list(zip([-1, -2], [1, 2]))
        for dim, obj, (lb, ub) in product(valid_dims, valid_objs, valid_bounds):
            stp_config = STPConfig(dim=dim, obj=obj, lb=lb, ub=ub)
            self.assertEqual(stp_config.dim, dim)
            self.assertEqual(stp_config.obj, obj)
            self.assertTrue(np.all(stp_config.lb == lb * np.ones(dim)))
            self.assertTrue(np.all(stp_config.ub == ub * np.ones(dim)))

    def test_invalid_required_inputs(self):
        for dim in [0.1, 1.1, -1, -2]:
            with self.assertRaises(ValueError, msg=f"while dim={dim}"):
                STPConfig(dim=dim, obj=1, lb=-1, ub=1)

        for obj in [0.1, 1.1, -1, -2]:
            with self.assertRaises(ValueError, msg=f"while obj={obj}"):
                STPConfig(dim=1, obj=obj, lb=-1, ub=1)

        invalid_bounds = [
            (1, 1), (2, -2), ([-2, 2], [2, -2]),  # invalid value
            ([1], [1]), ([-1], [1, 1]), ([-2, -2], [2]), ([-1, -1, -1], [1, 1, 1])  # invalid shape for dim=2
        ]
        for lb, ub in invalid_bounds:
            with self.assertRaises(ValueError, msg=f"while lb={lb}, ub={ub}"):
                STPConfig(dim=2, obj=1, lb=lb, ub=ub)

    def test_init_invalid_optionals(self):
        invalid_budgets = [
            (0, 10), (10, 0), (-10, 10), (-20, -10),  # fe_init or fe_max is non-positive
            (10, 9)  # fe_init > fe_max
        ]
        for fe_init, fe_max in invalid_budgets:
            with self.assertRaises(ValueError, msg=f"while fe_init={fe_init}, fe_max={fe_max}"):
                STPConfig(dim=2, obj=1, lb=0, ub=1, fe_init=fe_init, fe_max=fe_max)

        for npd in [-1, 0]:
            with self.assertRaises(ValueError, msg=f"while npd={npd}"):
                STPConfig(dim=2, obj=1, lb=0, ub=1, npd=npd)


class TestTransformerConfig(unittest.TestCase):

    def test_default(self):
        trans = TransformerConfig(dim=10)
        self.assertEqual(trans.dim, 10)
        self.assertTrue(np.all(trans.rotation == np.eye(10)))
        self.assertTrue(np.all(trans.rotation_inv == np.eye(10)))
        self.assertTrue(np.all(trans.shift == np.zeros(10)))

    def test_valid(self):
        dims = [2, 3, 4, 5]
        scales = [0.5, 1.0, 2.0, 5.0]
        for dim, scale in zip(dims, scales):
            trans = TransformerConfig(
                dim=dim,
                rotation=np.eye(dim) * scale,
                shift=np.ones(dim) * scale
            )
            self.assertTrue(np.all(trans.rotation == np.eye(dim) * scale))
            self.assertTrue(np.all(trans.shift == np.ones(dim) * scale))
            self.assertTrue(np.all(trans.rotation_inv @ trans.rotation == np.eye(dim)))

        for dim, shift in zip(dims, scales):
            trans = TransformerConfig(dim=dim, shift=shift*np.ones(dim))
            self.assertTrue(np.all(trans.shift == shift))

    def test_invalid(self):
        rot_dim_mismatch = {'dim': 2, 'rotation': np.eye(3)}
        shift_dim_mismatch = {'dim': 2, 'shift': np.ones(3)}

        self.assertRaises(ValueError, TransformerConfig, **rot_dim_mismatch)
        self.assertRaises(ValueError, TransformerConfig, **shift_dim_mismatch)


class TestFunctionInputs(unittest.TestCase):

    def test_valid_inputs(self):
        for valid_x in [1, [1, 2, 3], [1], (1,)]:
            fin = FunctionInputs(x=np.array(valid_x), dim=1)
            self.assertEqual(fin.x.ndim, 2)
            self.assertEqual(fin.x.shape[1], 1)

        for dim in [1, 2, 3, 5, 10, 20]:
            for valid_x in [np.ones(dim), np.ones((dim, 1)), np.ones((2, dim))]:
                fin = FunctionInputs(x=valid_x.squeeze(), dim=dim)
                self.assertEqual(fin.x.ndim, 2)
                self.assertEqual(fin.x.shape[1], dim)

    def test_invalid_inputs(self):
        for invalid_x in ['1', [[[1]]], [[1, 2], [3, 4]]]:
            with self.assertRaises(ValueError, msg=f"while x={invalid_x}"):
                if isinstance(invalid_x, list):
                    FunctionInputs(dim=1, x=np.array(invalid_x))
                else:
                    FunctionInputs(dim=1, x=invalid_x)


class TestPlottingArgs(unittest.TestCase):

    def test_valid_defaults(self):
        dim = 3
        lb = -1 * np.ones(dim)
        ub = np.ones(dim)
        plotting_args = PlottingArgs(
            dim=dim,
            dims=(0, 1),
            n_points=50,
            lb=lb,
            ub=ub,
            fixed=None,
        )
        self.assertEqual(plotting_args.dim, dim)
        self.assertEqual(plotting_args.dims, (0, 1))
        self.assertEqual(plotting_args.n_points, 50)
        np.testing.assert_array_equal(plotting_args.lb, lb)
        np.testing.assert_array_equal(plotting_args.ub, ub)
        np.testing.assert_array_equal(plotting_args.fixed, (lb + ub) / 2)

    def test_scalar_fixed_value(self):
        for fixed in [0.1, 0.2, 0.3, 0.4, 0.5]:
            plotting_args = PlottingArgs(
                dim=3,
                dims=(0, 1),
                n_points=50,
                lb=np.zeros(3),
                ub=np.ones(3),
                fixed=fixed,
            )
            self.assertTrue(np.all(plotting_args.fixed == fixed))

    def test_valid_selected_dims(self):
        valid_selected_dims = product([0, 1, 2, 3], [0, 1, 2, 3])
        for d1, d2 in valid_selected_dims:
            if d1 == d2:
                continue
            args = PlottingArgs(
                dim=4,
                dims=(d1, d2),
                fixed=None,
                lb=np.zeros(4),
                ub=np.ones(4),
                n_points=100,
            )
            self.assertEqual(args.dims[0], min(d1, d2), f"while dims={(d1, d2)}, args.dims=({args.dims})")
            self.assertEqual(args.dims[1], max(d1, d2), f"while dims={(d1, d2)}, args.dims=({args.dims})")

    def test_invalid_func_dim(self):
        invalid_func_dims = [1, -1, 0]
        for dim in invalid_func_dims:
            with self.assertRaises(ValueError, msg=f"while dim={dim}"):
                PlottingArgs(
                    dim=dim,
                    dims=(0, 1),
                    n_points=50,
                    lb=np.array([-1, -1]),
                    ub=np.array([1, 1]),
                    fixed=None,
                )

    def test_invalid_selected_dims(self):
        invalid_selected_dims = [
            (-1, 1), (2, 2), (1, 3),  # Out of range or not distinct
            (0, 1, 2)  # Too many selected dimensions
        ]
        for dims in invalid_selected_dims:
            with self.assertRaises(ValueError, msg=f"while dims={dims}"):
                PlottingArgs(
                    dim=3,
                    dims=dims,
                    n_points=50,
                    lb=np.array([-1, -1, -1]),
                    ub=np.array([1, 1, 1]),
                    fixed=None,
                )

    def test_invalid_n_points(self):
        for n_points in [5, 1500]:
            with self.assertWarns(UserWarning, msg=f"while n_points={n_points}"):
                PlottingArgs(
                    n_points=n_points,
                    dim=3,
                    dims=(0, 1),
                    lb=np.array([-1, -1, -1]),
                    ub=np.array([1, 1, 1]),
                    fixed=None,
                )

    def test_invalid_fixed(self):
        invalid_fixed = [
            np.array([2, 2, 2]),  # Out of bounds
            np.array([-2, -2, -2]),  # Out of bounds
            np.array([0, 0])  # Shape mismatch
        ]
        for fixed in invalid_fixed:
            with self.assertRaises(ValueError, msg=f"while fixed={fixed}"):
                PlottingArgs(
                    fixed=fixed,
                    dim=3,
                    dims=(0, 1),
                    n_points=50,
                    lb=np.array([-1, -1, -1]),
                    ub=np.array([1, 1, 1]),
                )
