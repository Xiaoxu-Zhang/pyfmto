import subprocess

import numpy as np
from itertools import product
from pathlib import Path

from pyfmto.problem import Solution
from pyfmto.experiment.utils import (
    RunSolutions, ReporterUtils, MetaData, MergedResults, ClientDataStatis
)
from pyfmto.utilities.schemas import STPConfig
from pyfmto.utilities import load_msgpack
from tests.helpers import PyfmtoTestCase
from tests.helpers.generators import ExpDataGenerator


def process_is_running(process: subprocess.Popen) -> bool:
    exit_code = process.poll()
    if exit_code is None:
        return True
    else:
        return False


def create_solution():
    solution = Solution(STPConfig(dim=2, obj=1, lb=[-1, -1], ub=[1, 1]))
    x = np.random.random((solution.fe_init, solution.dim))
    y = np.random.random((solution.fe_init, solution.obj))
    solution.append(x, y)
    return solution


class TestMetaData(PyfmtoTestCase):
    def setUp(self):
        self.save_sys_env()
        self.algs = ['ALG1', 'ALG2']
        self.probs = ['PROB1', 'PROB2']
        self.npd_names = ['NPD1', 'NPD2']
        self.n_task = 5
        self.n_run = 3
        self.generator = ExpDataGenerator(dim=10, lb=0, ub=1)

    def tearDown(self):
        self.delete('out')
        self.restore_sys_env()

    def test_empty(self):
        md = MetaData({}, 'P1', 'NIID', Path('tmp'))
        self.assertEqual(md.alg_num, 0)
        self.assertEqual(md.clt_num, 0)
        self.assertEqual(md.alg_names, [])
        self.assertEqual(md.clt_names, [])
        self.assertEqual(len(md), 0)
        self.assertEqual(md.dim, 0)
        with self.assertRaises(KeyError):
            _ = md['A1']
        with self.assertRaises(ValueError):
            _ = md.report_filename

    def test_properties(self):
        for prob, npd in product(self.probs, self.npd_names):
            md = self.generator.gen_metadata(algs=self.algs, prob=prob, npd=npd, n_tasks=5, n_runs=3)

            self.assertEqual(len(md), len(self.algs))
            self.assertEqual(md.alg_num, len(self.algs))
            self.assertEqual(md.clt_num, self.n_task)
            self.assertEqual(md.alg_names, self.algs)
            self.assertEqual(md.dim, 10)
            self.assertEqual(md.num_runs, self.n_run)
            self.assertEqual(md.clt_names, [f'Client {i+1:>02d}' for i in range(self.n_task)])
            for alg in self.algs:
                self.assertIsInstance(md[alg], MergedResults)
            self.assertIsInstance(md.items(), list)
            self.assertEqual(len(md.items()), len(self.algs))


class TestMergedResults(PyfmtoTestCase):
    def setUp(self):
        self.generator = ExpDataGenerator(dim=10, lb=0, ub=1)

    def test_methods(self):
        mr = self.generator.gen_merged_data(n_tasks=5, n_runs=3)
        self.assertEqual(len(mr.items()), 5)
        self.assertIsInstance(mr.get_statis(mr.sorted_names[0]), ClientDataStatis)

    def test_empty(self):
        with self.assertRaises(ValueError):
            MergedResults([])


class TestClientDataStatis(PyfmtoTestCase):
    def setUp(self):
        self.conf = STPConfig(dim=10, obj=1, lb=0, ub=1)
        self.generator = ExpDataGenerator(dim=10, lb=0, ub=1)

    def test_init_empty(self):
        with self.assertRaises(ValueError):
            ClientDataStatis([])

    def test_properties(self):
        cds = ClientDataStatis(self.generator.gen_solutions(n_solutions=10))
        self.assertTrue(np.all(cds.lb == self.conf.lb))
        self.assertTrue(np.all(cds.ub == self.conf.ub))
        self.assertEqual(cds.fe_init, self.conf.fe_init)
        self.assertEqual(cds.fe_max, self.conf.fe_max)
        self.assertEqual(cds.x_init.shape[0], cds.fe_init * 10)
        self.assertEqual(cds.y_init.shape[0], cds.fe_init * 10)
        self.assertEqual(cds.x_alg.shape[0], (cds.fe_max - cds.fe_init) * 10)
        self.assertTrue(cds.is_known_optimal)
        sub_test_cases = [cds.y_dec_statis, cds.y_inc_statis, cds.y_dec_log_statis, cds.y_inc_log_statis]
        sub_test_names = ['y_dec_statis', 'y_inc_statis', 'y_dec_log_statis', 'y_inc_log_statis']
        for statis, name in zip(sub_test_cases, sub_test_names):
            with self.subTest(statis):
                if 'log' in name:
                    self.assertTrue(np.all(statis.mean <= 0), f"on {name} mean is\n{statis.mean}")
                else:
                    self.assertTrue(np.all(statis.mean >= 0), f"on {name} mean is\n{statis.mean}")
                self.assertEqual(statis.mean.shape[0], self.conf.fe_max)
                self.assertEqual(statis.std.shape[0], self.conf.fe_max)
                self.assertTrue(np.all(statis.std > 0))
                self.assertEqual(statis.opt.shape[0], 10)


class TestReporterUtils(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        self.utils = ReporterUtils

    def tearDown(self):
        self.delete('out')
        self.restore_sys_env()

    def test_load_runs_data(self):
        self.assertEqual(self.utils.load_runs_data(Path('out')), [])

    def test_get_t_test_suffix(self):
        test_cases = [
            {
                "opt_list1": [1, 2, 3],
                "opt_list2": [4, 5, 6],
                "mean1": 2.0,
                "mean2": 5.0,
                "pvalue": 0.05,
                "expected": "+"  # diff > 0 and p <= pvalue
            },
            {
                "opt_list1": [4, 5, 6],
                "opt_list2": [1, 2, 3],
                "mean1": 5.0,
                "mean2": 2.0,
                "pvalue": 0.05,
                "expected": "-"  # diff <= 0 and p <= pvalue
            },
            {
                "opt_list1": [1, 2, 3],
                "opt_list2": [1, 2, 3],
                "mean1": 2.0,
                "mean2": 2.0,
                "pvalue": 0.05,
                "expected": "≈"  # p > pvalue
            },
            {
                "opt_list1": [10, 20, 30],
                "opt_list2": [15, 25, 35],
                "mean1": 20.0,
                "mean2": 25.0,
                "pvalue": 0.1,
                "expected": "≈"  # p > pvalue
            }
        ]
        for case in test_cases:
            kwargs = case.copy()
            expected = kwargs.pop('expected')
            self.assertEqual(self.utils.get_t_test_suffix(**kwargs), expected)

    def test_all_rows_equal(self):
        col_title = ["Col1", "Col2", "Col3"]
        row_title = ["Row1", "Row2", "Row3"]

        valid_data = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        ReporterUtils.check_rows(valid_data, col_title, row_title, f'Using valid data {valid_data}')

        rows_different = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        invalid_value = [["a", "b", "c"], [1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            ReporterUtils.check_rows(invalid_value, col_title, row_title,
                                     f'Data with different rows {invalid_value}')
        with self.assertRaises(ValueError):
            ReporterUtils.check_rows(rows_different, col_title, row_title,
                                     f'Data with invalid values {rows_different}')

    def test_find_grid_shape(self):
        test_cases = [
            (1, (1, 1)),
            (4, (2, 2)),
            (6, (3, 2)),
            (10, (4, 3)),
            (12, (4, 3)),
            (18, (5, 4)),
            (25, (5, 5)),
            (30, (6, 5)),
            (50, (8, 7)),
            (60, (8, 8)),
        ]

        for size, expected in test_cases:
            with self.subTest(size=size):
                self.assertEqual(ReporterUtils.find_grid_shape(size), expected)

        with self.assertRaises(ValueError):
            ReporterUtils.find_grid_shape(0)


class TestRunSolutions(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        self.tmp_dir = Path('temp_dir_for_test')
        self.tmp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        self.delete(self.tmp_dir)
        self.restore_sys_env()

    def test_initialization(self):
        rs = RunSolutions()
        self.assertEqual(rs.num_clients, 0)
        self.assertEqual(rs.sorted_ids, [])

    def test_update_and_get_solution(self):
        rs = RunSolutions()
        solution = create_solution()

        rs.update(1, solution)

        self.assertEqual(rs.num_clients, 1)
        self.assertEqual(rs.sorted_ids, [1])

        retrieved = rs[1]
        self.assertIsInstance(retrieved, Solution)
        self.assertEqual(retrieved.dim, solution.dim)
        self.assertEqual(retrieved.obj, solution.obj)
        self.assertEqual(retrieved.size, solution.size)

        self.assertIsInstance(rs.solutions, list)
        self.assertEqual(len(rs.solutions), 1)

    def test_get_nonexistent_client_raises_error(self):
        rs = RunSolutions()
        with self.assertRaises(KeyError):
            print(rs[99])

    def test_save_empty(self):
        rs = RunSolutions()
        self.assertRaises(ValueError, rs.to_msgpack, self.tmp_dir / 'test.msgpack')

    def test_save_and_load(self):
        rs1 = RunSolutions()
        solution = create_solution()

        rs1.update(1, solution)
        rs1.update(2, solution)
        rs1.to_msgpack(self.tmp_dir / 'test.msgpack')
        rs2 = RunSolutions(load_msgpack(self.tmp_dir / 'test.msgpack'))

        self.assertEqual(rs2.num_clients, rs1.num_clients)
        self.assertEqual(rs2.sorted_ids, rs1.sorted_ids)

        for cid in rs1.sorted_ids:
            s1 = rs1[cid]
            s2 = rs2[cid]

            np.testing.assert_array_equal(s1.x, s2.x)
            np.testing.assert_array_equal(s1.y, s2.y)
            self.assertEqual(s1.dim, s2.dim)
            self.assertEqual(s1.obj, s2.obj)
            self.assertEqual(s1.fe_init, s2.fe_init)
