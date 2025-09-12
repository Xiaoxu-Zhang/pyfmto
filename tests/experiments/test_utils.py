import subprocess
import numpy as np
import shutil
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
from pyfmto import framework as fw, load_problem
from pyfmto.framework import Client, Server
from pyfmto.problems import Solution
from pyfmto.experiments.utils import RunSolutions, LauncherUtils, ReporterUtils, list_algorithms, load_algorithm
from pyfmto.utilities.schemas import LauncherConfig, STPConfig
from pyfmto.utilities import load_msgpack
from tests.experiments import export_alg_template
from tests.framework import OnlineServer


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


class TestReporterUtils(unittest.TestCase):

    def setUp(self):
        self.utils = ReporterUtils

    def tearDown(self):
        shutil.rmtree(Path('out'), ignore_errors=True)

    def test_get_runs_data(self):
        prob = load_problem('tetci2019')
        all_runs = []
        for runs in range(1, 6):
            run_data = RunSolutions()
            for p in prob[:5]:
                run_data.update(p.id, p.solutions)
            all_runs.append(run_data)
        res = self.utils.get_runs_data(1, all_runs)
        self.assertEqual(len(res['runs_x']), 5)
        self.assertEqual(len(res['runs_y']), 5)
        self.assertEqual(res['fe_init'], prob[0].fe_init)
        self.assertEqual(res['fe_max'], all_runs[0].get_solutions(1).size)

    def test_calculate_statistics(self):
        test_cases = [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1e-25, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ]

        for i, data in enumerate(test_cases):
            with self.subTest(test_case=i + 1):
                result = ReporterUtils.calculate_statistics(data)
                rows, cols = np.array(data).shape
                self.assertEqual(result.mean_orig.shape, (cols, ))
                self.assertEqual(result.std_orig.shape, (cols, ))
                self.assertEqual(result.se_orig.shape, (cols, ))
                self.assertEqual(result.opt_orig.shape, (rows, ))
                self.assertEqual(result.opt_log.shape, (rows, ))
                self.assertFalse(result.is_known_optimal)

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

    def test_parse_reporter_config(self):
        config = {
            'algorithms': [['alg1', 'alg2'], ['alg3', 'alg4']],
            'problems': ['prob1', 'prob2']
        }
        prob_conf = {
            'prob1': {'src_problem': 'src1', 'np_per_dim': 2},
            'prob2': {'np_per_dim': 3}
        }
        result = self.utils.parse_reporter_config(config, prob_conf)

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


class TestLauncherUtils(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        self.tmp_server = Path('temp_server.py')
        self.tmp_setting = Path('settings.yaml')
        self.tmp_alg_dir = Path('algorithms')
        self.utils = LauncherUtils
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir()

        self.run_settings_ok = {
            'launcher': {
                'algorithms': ['FMTBO'],
                'problems': ['CEC2022']
            },
            'reporter': {
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

    def test_start_server(self):
        with self.utils.running_server(OnlineServer) as s:
            self.assertTrue(process_is_running(s))
        self.assertFalse(process_is_running(s))

    def test_terminate_popen_normal(self):
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_process.wait = Mock()

        LauncherUtils.terminate_popen(mock_process)

        mock_process.stdout.close.assert_called_once()
        mock_process.stderr.close.assert_called_once()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

        mock_process.kill.assert_not_called()

    def test_terminate_popen_timeout(self):
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()

        def wait_side_effect(*args, **kwargs):
            if not hasattr(wait_side_effect, "called"):
                wait_side_effect.called = True
                raise subprocess.TimeoutExpired(cmd="cmd", timeout=5)
            else:
                return None

        mock_process.wait = Mock(side_effect=wait_side_effect)

        LauncherUtils.terminate_popen(mock_process)

        mock_process.stdout.close.assert_called_once()
        mock_process.stderr.close.assert_called_once()
        mock_process.terminate.assert_called_once()
        self.assertEqual(mock_process.wait.call_count, 2)
        mock_process.wait.assert_any_call(timeout=5)
        mock_process.kill.assert_called_once()

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
        res_root1 = self.utils.gen_path(alg, prob, kwargs1)
        res_root2 = self.utils.gen_path(alg, prob, kwargs2)
        self.assertEqual(res_root1, Path('out', 'results', alg, f"{prob}_2D", "IID"))
        self.assertEqual(res_root2, Path('out', 'results', alg, f"{prob}_2D", "NIID2"))

    def test_cross_platform_tools(self):
        with patch('os.system') as mock_system:
            with patch('platform.system', return_value="Windows"):
                self.utils.kill_server()
                mock_system.assert_called_once_with("taskkill /f /im AlgServer.exe")
                mock_system.reset_mock()

            with patch('platform.system', return_value="Linux"):
                self.utils.kill_server()
                mock_system.assert_called_once_with("pkill -f AlgServer")
                mock_system.reset_mock()

    def test_combine_args_no_list(self):
        args = {
            'problem1': {'param1': 'value1', 'param2': 'value2'}
        }
        result = self.utils.combine_args(args)
        expected = [('problem1', {'param1': 'value1', 'param2': 'value2'})]
        self.assertEqual(result, expected)

    def test_combine_args_with_list(self):
        args = {
            'problem1': {'param1': 'value1', 'param2': [1, 2]}
        }
        result = self.utils.combine_args(args)
        expected = [
            ('problem1', {'param1': 'value1', 'param2': 1}),
            ('problem1', {'param1': 'value1', 'param2': 2})
        ]
        self.assertEqual(result, expected)

    def test_combine_args_multiple_lists(self):
        args = {
            'problem1': {'param1': [1, 2], 'param2': ['a', 'b']}
        }
        result = self.utils.combine_args(args)
        expected = [
            ('problem1', {'param1': 1, 'param2': 'a'}),
            ('problem1', {'param1': 1, 'param2': 'b'}),
            ('problem1', {'param1': 2, 'param2': 'a'}),
            ('problem1', {'param1': 2, 'param2': 'b'})
        ]
        self.assertEqual(result, expected)

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
        result = self.utils.gen_exp_combinations(launcher_conf, alg_conf, prob_conf)

        expected = [
            ('alg1', {'alg_param': 1}, 'prob1', {'prob_param': 'a'}),
            ('alg1', {'alg_param': 1}, 'prob2', {'prob_param': 'b'}),
            ('alg2', {'alg_param': 2}, 'prob1', {'prob_param': 'a'}),
            ('alg2', {'alg_param': 2}, 'prob2', {'prob_param': 'b'})
        ]
        self.assertEqual(result, expected)


class TestRunSolutions(unittest.TestCase):

    def setUp(self):
        self.tmp = Path('tmp')
        if not self.tmp.exists():
            self.tmp.mkdir()

    def tearDown(self):
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

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

        retrieved = rs.get_solutions(1)
        self.assertIsInstance(retrieved, Solution)
        self.assertEqual(retrieved.dim, solution.dim)
        self.assertEqual(retrieved.obj, solution.obj)
        self.assertEqual(retrieved.size, solution.size)

        self.assertIsInstance(rs.solutions, list)
        self.assertEqual(len(rs.solutions), 1)

    def test_get_nonexistent_client_raises_error(self):
        rs = RunSolutions()
        with self.assertRaises(KeyError):
            rs.get_solutions(99)

    def test_save_empty(self):
        rs = RunSolutions()
        self.assertRaises(ValueError, rs.to_msgpack, self.tmp / 'test.msgpack')

    def test_save_and_load(self):
        rs1 = RunSolutions()
        solution = create_solution()

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
        solution = create_solution()
        rs.update(1, solution)

        rs.clear()
        self.assertEqual(rs.num_clients, 0)
        self.assertEqual(rs.sorted_ids, [])


class TestExportTools(unittest.TestCase):
    def setUp(self):
        self.files = ['run.py', 'config.yaml', 'report.py']
        self.alg_dir = Path('algorithms')
        self.conf = Path('config.yaml')

    def tearDown(self):
        shutil.rmtree(self.alg_dir, ignore_errors=True)
        Path('run.py').unlink(missing_ok=True)
        for filename in self.files:
            Path(filename).unlink(missing_ok=True)

    def test_export_to_new(self):
        funcs = [
            fw.export_launcher_config,
            fw.export_reporter_config,
            fw.export_problem_config,
        ]

        # each func repeat twice to cover file exists case
        tmp_files = [f() for f in funcs + funcs]
        for p in tmp_files:
            self.assertTrue(p.exists())
        for p in tmp_files:
            p.unlink()

    def test_export_invalid_config(self):
        fw.export_algorithm_config(algs=('INVALID', ))
        fw.export_problem_config(probs=('INVALID', ))


class TestOtherUtils(unittest.TestCase):

    def setUp(self):
        self.alg_dir = Path('algorithms')
        self.alg_dir.mkdir(parents=True, exist_ok=True)
        export_alg_template('ALG1')
        export_alg_template('ALG2')
        empty = self.alg_dir / 'INVALID'
        empty.mkdir()

    def tearDown(self):
        shutil.rmtree(self.alg_dir, ignore_errors=True)

    def test_list_algorithms(self):
        list_algorithms(print_it=True)

    def test_load_algorithm(self):
        res = load_algorithm('ALG1')
        self.assertEqual(res.name, 'ALG1')
        self.assertTrue(issubclass(res.client, Client))
        self.assertTrue(issubclass(res.server, Server))

    def test_load_invalid_algorithm(self):
        with self.assertRaises(RuntimeError):
            load_algorithm('INVALID')
