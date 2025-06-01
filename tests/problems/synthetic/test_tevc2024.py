#!usr/bin/env/ python
# -*- coding:utf-8 -*-

import unittest

from pyfmto.problems import load_problem

_ORIGINAL_PROBLEMS = ["Griewank", "Rastrigin", "Ackley", "Schwefel", "Sphere", "Rosenbrock", "Weierstrass", "Ellipsoid"]


class TestTevc2024(unittest.TestCase):
    def test_init(self):
        problems = load_problem('tevc2024', _init_solutions=False)
        prob = problems[0]
        self.assertEqual(prob.dim, 10)
        self.assertEqual(prob.obj, 1)
        self.assertEqual(prob.name, 'Ackley')

    def test_rasis(self):
        raise_type = {'src_problem': 5}
        raise_value = {'src_problem': 'Ackleys'}
        self.assertRaises(TypeError, load_problem, 'tevc2024', **raise_type)
        self.assertRaises(ValueError, load_problem, 'tevc2024', **raise_value)

    def test_not_rasis(self):
        problems = load_problem('tevc2024')
        self.assertTrue(problems.problem_name in str(problems))
