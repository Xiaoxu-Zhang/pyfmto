#!usr/bin/env/ python
# -*- coding:utf-8 -*-

import unittest
from pyfmto.problems import load_problem


class TestSvmLandmine(unittest.TestCase):
    def test_init_svm_landmine(self):
        problem = load_problem('svm_landmine')
        self.assertEqual(29, len(problem))
        task = problem[0]
        x_test = task.random_uniform_x(10)
        y_test = task.evaluate(x_test)
        _ = task.evaluate(x_test[0])
        self.assertEqual(10, y_test.shape[0])
        self.assertEqual(1, y_test.shape[1])
        _ = str(problem)
