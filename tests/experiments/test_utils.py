import numpy as np
import shutil
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch

from pyfmto.experiments.utils import (
    gen_path,
    check_path,
    load_results,
    save_results,
    clear_console,
    RunSolutions
)
from pyfmto.problems import Solution

TMP_ALG = """
class TmpClient:
    pass
    
class TmpServer:
    pass
"""

TMP_INIT = """
from .tmp_alg import TmpClient, TmpServer
"""

SETTINGS_YML = """
launcher:
  repeat: 3         # number of runs repeating
  backup: True      # backup log file to results directory
  dir: out/results  # dir of results
  save: True        # save results
  seed: 42          # random seed
  algorithms: [FDEMD, FMTBO]
  problems: [cec2022]
reporter:
  results: ~
  algorithms:
    - [FMTBO, FDEMD]
  problems: [CEC2022]
  np_per_dim: [1, 2]

"""


class TestExperimentUtils(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        self.tmp_server = Path('temp_server.py')
        self.tmp_setting = Path('settings.yaml')
        self.tmp_alg_dir = Path('algorithms')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir()

        self.run_settings_ok = {
            'launcher': {
                'algorithms': ['FMTBO'],
                'problems': ['CEC2022']
            },
            'reporter':{
                'algorithms': [['FMTBO', 'FDEMD']],
                'np_per_dim': [1, 2],
                'problems': ['CEC2022'],
            }
        }
        with open(self.tmp_setting, 'w') as f:
            yaml.dump(self.run_settings_ok, f)

    def tearDown(self):
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir)
        if self.tmp_server.exists():
            self.tmp_server.unlink()
        if self.tmp_setting.exists():
            self.tmp_setting.unlink()
        if self.tmp_alg_dir.exists():
            shutil.rmtree('algorithms')

    def test_gen_path(self):
        alg = 'ALG'
        prob = 'PROB'
        kwargs1 = {
            'np_per_dim': 1,
            'dim': 2,
        }
        kwargs2 = {
            'np_per_dim': 2,
            'dim': 2,
        }
        res_root1 = gen_path(alg, prob, kwargs1)
        res_root2 = gen_path(alg, prob, kwargs2)
        self.assertEqual(res_root1, Path('out', 'results', alg, f"{prob}_2D", "IID"))
        self.assertEqual(res_root2, Path('out', 'results', alg, f"{prob}_2D", "NIID2"))

    def test_check_path(self):
        file_num = check_path(self.tmp_dir / 'test_path')
        self.assertEqual(file_num, 0)
        with open(self.tmp_dir / 'test_path' / 'test_file.txt', 'w') as f:
            f.write('test content')
        file_num = check_path(self.tmp_dir / 'test_path')
        self.assertEqual(file_num, 1)

    def test_clear_console(self):
        with patch('os.system') as mock_system:
            with patch('os.name', 'posix'):
                clear_console()
                mock_system.assert_called_once_with('clear')
                mock_system.reset_mock()

            with patch('os.name', 'nt'):
                clear_console()
                mock_system.assert_called_once_with('cls')


class TestRunSolutions(unittest.TestCase):

    def setUp(self):
        self.tmp = Path('tmp')
        if not self.tmp.exists():
            self.tmp.mkdir()

    def tearDown(self):
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    @staticmethod
    def _create_solution():
        solution = Solution()
        dim, obj = 2, 1

        x = np.random.random((5, dim))
        y = np.random.random((5, obj))
        solution.append(x, y)
        return solution

    def test_initialization(self):
        rs = RunSolutions()
        self.assertEqual(rs.num_clients, 0)
        self.assertEqual(rs.sorted_ids, [])

    def test_update_and_get_solution(self):
        rs = RunSolutions()
        solution = self._create_solution()

        rs.update(1, solution)

        self.assertEqual(rs.num_clients, 1)
        self.assertEqual(rs.sorted_ids, [1])

        retrieved = rs.get_solutions(1)
        self.assertIsInstance(retrieved, Solution)
        self.assertEqual(retrieved.dim, 2)
        self.assertEqual(retrieved.obj, 1)
        self.assertEqual(retrieved.size, 5)

    def test_get_multiple_solutions(self):
        rs = RunSolutions()
        solution = self._create_solution()

        rs.update(1, solution)
        rs.update(2, solution)

        retrieved = rs.get_solutions([1, 2])
        self.assertIsInstance(retrieved, dict)
        self.assertIn(1, retrieved)
        self.assertIn(2, retrieved)

    def test_get_nonexistent_client_raises_error(self):
        rs = RunSolutions()
        with self.assertRaises(KeyError):
            rs.get_solutions(99)

    def test_save_load_results(self):
        data = [(i, self._create_solution()) for i in range(5)]
        save_results(data, self.tmp, 1)
        expect_file = self.tmp / 'Run 1.msgpack'
        self.assertTrue(expect_file.exists())
        load_data = load_results(expect_file)
        self.assertIsInstance(load_data, RunSolutions)
        self.assertEqual(load_data.num_clients, 5)
        empty_data = []
        save_results(empty_data, self.tmp, 2) # don't save empty data
        expect_file = self.tmp / 'Run 2.msgpack'
        self.assertEqual(False, expect_file.exists())

    def test_save_and_load(self):
        rs1 = RunSolutions()
        solution = self._create_solution()

        rs1.update(1, solution)
        rs1.update(2, solution)
        rs1.to_msgpack(self.tmp / 'test.msgpack')
        rs2 = load_results(self.tmp / 'test.msgpack')

        self.assertEqual(rs2.num_clients, rs1.num_clients)
        self.assertEqual(rs2.sorted_ids, rs1.sorted_ids)

        for cid in rs1.sorted_ids:
            s1 = rs1.get_solutions(cid)
            s2 = rs2.get_solutions(cid)

            np.testing.assert_array_equal(s1.x, s2.x)
            np.testing.assert_array_equal(s1.y, s2.y)
            self.assertEqual(s1.dim, s2.dim)
            self.assertEqual(s1.obj, s2.obj)
            self.assertEqual(s1.fe_init, s2.fe_init)

    def test_clear_resets_state(self):
        rs = RunSolutions()
        solution = self._create_solution()
        rs.update(1, solution)

        rs.clear()
        self.assertEqual(rs.num_clients, 0)
        self.assertEqual(rs.sorted_ids, [])
