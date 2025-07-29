import numpy as np
import shutil
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch

from pyfmto.experiments import Statistics
from pyfmto.problems import Solution
from pyfmto.experiments.utils import (
    gen_path, clear_console,
    kill_server, gen_exp_combinations, combine_args, parse_reporter_config,
    RunSolutions
)
from pyfmto.utilities.schemas import LauncherConfig
from pyfmto.utilities import load_msgpack

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

        self.assertIsInstance(rs.solutions, list)
        self.assertEqual(len(rs.solutions), 1)

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

    def test_save_empty(self):
        rs = RunSolutions()
        self.assertRaises(ValueError, rs.to_msgpack, self.tmp/'test.msgpack')

    def test_save_and_load(self):
        rs1 = RunSolutions()
        solution = self._create_solution()

        rs1.update(1, solution)
        rs1.update(2, solution)
        rs1.to_msgpack(self.tmp / 'test.msgpack')
        rs2 = RunSolutions(load_msgpack(self.tmp / 'test.msgpack'))

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


class TestStatistics(unittest.TestCase):

    def test_init(self):
        sta = Statistics(
            mean_orig=np.array([1, 2]),
            mean_log=np.array([3, 4]),
            std_orig=np.array([5, 6]),
            std_log=np.array([7, 8]),
            se_orig=np.array([9, 10]),
            se_log=np.array([11, 12]),
            opt_orig=np.array([13, 14]),
            opt_log=np.array([15, 16])
        )
        sta.fe_init = 10
        sta.fe_max = 20
        sta.x = np.array([17, 18])
        sta.x_global = np.array([19, 20])
        sta.y_global = np.array([21, 22])

        self.assertTrue(np.all(sta.mean_orig==np.array([1, 2])))
        self.assertTrue(np.all(sta.mean_log==np.array([3, 4])))
        self.assertTrue(np.all(sta.std_orig==np.array([5, 6])))
        self.assertTrue(np.all(sta.std_log==np.array([7, 8])))
        self.assertTrue(np.all(sta.se_orig==np.array([9, 10])))
        self.assertTrue(np.all(sta.se_log==np.array([11, 12])))
        self.assertTrue(np.all(sta.opt_orig==np.array([13, 14])))
        self.assertTrue(np.all(sta.opt_log==np.array([15, 16])))
        self.assertTrue(np.all(sta.x==np.array([17, 18])))
        self.assertTrue(np.all(sta.x_global==np.array([19, 20])))
        self.assertTrue(np.all(sta.y_global==np.array([21, 22])))
        self.assertEqual(sta.fe_init, 10)
        self.assertEqual(sta.fe_max, 20)


class TestKillServer(unittest.TestCase):

    @patch('os.name', 'win32')
    @patch('os.system')
    def test_kill_server_windows(self, mock_system):
        kill_server()
        mock_system.assert_called_once_with("taskkill /f /im AlgServer.exe")

    @patch('os.name', 'posix')
    @patch('os.system')
    def test_kill_server_posix(self, mock_system):
        kill_server()
        mock_system.assert_called_once_with("pkill -f AlgServer")


class TestCombineArgs(unittest.TestCase):

    def test_combine_args_no_list(self):
        args = {
            'problem1': {'param1': 'value1', 'param2': 'value2'}
        }
        result = combine_args(args)
        expected = [('problem1', {'param1': 'value1', 'param2': 'value2'})]
        self.assertEqual(result, expected)

    def test_combine_args_with_list(self):
        args = {
            'problem1': {'param1': 'value1', 'param2': [1, 2]}
        }
        result = combine_args(args)
        expected = [
            ('problem1', {'param1': 'value1', 'param2': 1}),
            ('problem1', {'param1': 'value1', 'param2': 2})
        ]
        self.assertEqual(result, expected)

    def test_combine_args_multiple_lists(self):
        args = {
            'problem1': {'param1': [1, 2], 'param2': ['a', 'b']}
        }
        result = combine_args(args)
        expected = [
            ('problem1', {'param1': 1, 'param2': 'a'}),
            ('problem1', {'param1': 1, 'param2': 'b'}),
            ('problem1', {'param1': 2, 'param2': 'a'}),
            ('problem1', {'param1': 2, 'param2': 'b'})
        ]
        self.assertEqual(result, expected)


class TestGenExpCombinations(unittest.TestCase):

    def test_gen_exp_combinations(self):
        launcher_conf = LauncherConfig(
            algorithms=['alg1', 'alg2'],
            problems=['prob1', 'prob2']
        )
        alg_conf = {
            'alg1': {'alg_param': 1},
            'alg2': {'alg_param': 2}
        }
        prob_conf = {
            'prob1': {'prob_param': 'a'},
            'prob2': {'prob_param': 'b'}
        }
        result = gen_exp_combinations(launcher_conf, alg_conf, prob_conf)

        expected = [
            ('alg1', {'alg_param': 1}, 'prob1', {'prob_param': 'a'}),
            ('alg1', {'alg_param': 1}, 'prob2', {'prob_param': 'b'}),
            ('alg2', {'alg_param': 2}, 'prob1', {'prob_param': 'a'}),
            ('alg2', {'alg_param': 2}, 'prob2', {'prob_param': 'b'})
        ]
        self.assertEqual(result, expected)


class TestParseReporterConfig(unittest.TestCase):

    def test_parse_reporter_config(self):
        config = {
            'algorithms': [['alg1', 'alg2'], ['alg3', 'alg4']],
            'problems': ['prob1', 'prob2']
        }
        prob_conf = {
            'prob1': {'src_problem': 'src1', 'np_per_dim': 2},
            'prob2': {'np_per_dim': 3}
        }
        result = parse_reporter_config(config, prob_conf)

        expected_analysis_comb = [
            (['alg1', 'alg2'], 'PROB1-src1', 2),
            (['alg1', 'alg2'], 'PROB2', 3),
            (['alg3', 'alg4'], 'PROB1-src1', 2),
            (['alg3', 'alg4'], 'PROB2', 3)
        ]

        expected_initialize_comb = [
            ('alg1', 'PROB1-src1', 2),
            ('alg1', 'PROB2', 3),
            ('alg2', 'PROB1-src1', 2),
            ('alg2', 'PROB2', 3),
            ('alg3', 'PROB1-src1', 2),
            ('alg3', 'PROB2', 3),
            ('alg4', 'PROB1-src1', 2),
            ('alg4', 'PROB2', 3)
        ]

        self.assertEqual(result['results'], 'out/results')
        self.assertEqual(result['analysis_comb'], expected_analysis_comb)
        self.assertEqual(result['initialize_comb'], expected_initialize_comb)
