import unittest
import numpy as np

from pyfmto.problems.problem import SingleTaskProblem, check_and_transform
from pyfmto.problems.benchmarks import (
    Griewank as _Griewank,
    Rastrigin as _Rastrigin,
    Ackley as _Ackley,
    Schwefel as _Schwefel,
    Sphere as _Sphere,
    Rosenbrock as _Rosenbrock,
    Weierstrass as _Weierstrass,
    Ellipsoid as _Ellipsoid
)


class BatchEval(SingleTaskProblem):
    def __init__(self, dim, obj, x_lb, x_ub):
        super().__init__(dim=dim, obj=obj, x_lb=x_lb, x_ub=x_ub)

    @check_and_transform()
    def evaluate(self, x):
        res = []
        for xi in x:
            res.append(self._eval_one(xi))
        return np.array(res).reshape(-1, 1)

    def _eval_one(self, x):
        pass


class OrigGriewank(BatchEval):

    def __init__(self, x_lb=-600, x_ub=600, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        F1 = 0
        F2 = 1
        d_x = len(x)
        for i in range(0, d_x):
            # z = x[i] - griewank[i]
            z = x[i]
            F1 = F1 + (z ** 2 / 4000)
            F2 = F2 * (np.cos(z / np.sqrt(i + 1)))
        return F1 - F2 + 1


class OrigRastrigin(BatchEval):

    def __init__(self, x_lb=-5, x_ub=5, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        F = 0
        d_x = len(x)
        for i in range(0, d_x):
            z = x[i]
            F = F + (z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
        return F


class OrigAckley(BatchEval):

    def __init__(self, x_lb=-32.768, x_ub=32.768, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        sum1 = 0
        sum2 = 0
        d_x = len(x)
        for i in range(0, d_x):
            z = x[i]
            sum1 = sum1 + z ** 2
            sum2 = sum2 + np.cos(2 * np.pi * z)
        out = -20 * np.exp(-0.2 * np.sqrt(sum1 / d_x)) - np.exp(sum2 / d_x) + 20 + np.e
        return out


class OrigSchwefel(BatchEval):

    def __init__(self, x_lb=-500, x_ub=500, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        out = 0
        d_x = len(x)
        for i in range(d_x):
            out += x[i] * np.sin(np.sqrt(abs(x[i])))
        return 418.9829 * d_x - out


class OrigSphere(BatchEval):

    def __init__(self, x_lb=-100, x_ub=100, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        return x.dot(x)


class OrigRosenbrock(BatchEval):
    def __init__(self, x_lb=-2.048, x_ub=2.048, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        out = 0
        d_x = len(x)
        for i in range(0, d_x - 1):
            tmp = 100 * np.power(x[i + 1] ** 2 - x[i], 2) + np.power(x[i] - 1, 2)
            out += tmp
        return out


class OrigWeierstrass(BatchEval):

    def __init__(self, x_lb=-0.5, x_ub=0.5, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        D = len(x)
        a = 0.5
        b = 3
        kmax = 21
        obj = 0
        for i in range(D):
            for k in range(kmax):
                obj += a ** k * np.cos(2 * np.pi * b ** k * (x[i] + 0.5))
        for k in range(kmax):
            obj -= D * a ** k * np.cos(2 * np.pi * b ** k * 0.5)
        return obj


class OrigEllipsoid(BatchEval):

    def __init__(self, x_lb=-5.12, x_ub=5.12, d=10):
        super().__init__(dim=d, obj=1, x_lb=x_lb, x_ub=x_ub)

    def _eval_one(self, x):
        out = 0.0
        d = len(x)
        d_L = [i for i in range(1, d + 1)]
        for i_ in d_L:
            out += i_ * x[i_ - 1] ** 2
        return out


curr_impl: list[SingleTaskProblem] = [_Griewank(), _Rastrigin(), _Ackley(), _Schwefel(), _Sphere(), _Rosenbrock(),
                                      _Weierstrass(), _Ellipsoid()]
orig_impl: list[SingleTaskProblem] = [OrigGriewank(), OrigRastrigin(), OrigAckley(), OrigSchwefel(), OrigSphere(),
                                      OrigRosenbrock(), OrigWeierstrass(), OrigEllipsoid()]


class TestCorrectness(unittest.TestCase):

    def test_properties(self):
        for impl, orig in zip(curr_impl, orig_impl):
            self.assertEqual(impl.dim, orig.dim)
            self.assertEqual(impl.obj, orig.obj)
            self.assertTrue(np.all(impl.x_lb == orig.x_lb))
            self.assertTrue(np.all(impl.x_ub == orig.x_ub))
            x = impl.random_uniform_x(size=100)
            y_impl = impl.evaluate(x)
            y_orig = orig.evaluate(x)
            self.assertLess(np.mean(np.abs(y_impl - y_orig)), 1e-6)
