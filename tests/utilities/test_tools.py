import subprocess
import unittest
from unittest.mock import patch, Mock

from pyfmto.utilities import (
    colored,
    show_in_table,
    update_kwargs,
    clear_console,
    titled_tabulate,
    terminate_popen,
    tabulate_formats,
)

from tests.helpers import remove_temp_files


class TestTools(unittest.TestCase):

    def tearDown(self):
        remove_temp_files()

    def test_cross_platform_tools(self):
        with patch('os.system') as mock_system:

            with patch('platform.system', return_value="Windows"):
                clear_console()
                mock_system.assert_called_once_with('cls')
                mock_system.reset_mock()

            with patch('platform.system', return_value="Linux"):
                clear_console()
                mock_system.assert_called_once_with('clear')
                mock_system.reset_mock()

    def test_colored(self):
        text = "test"
        color_text = colored(text, 'green')
        self.assertIn('\033[32m', color_text)
        self.assertIn(text, color_text)
        self.assertRaises(ValueError, colored, text, 'invalid_color')

    def test_show_in_table(self):
        settings = {
            "verbose": True,
            "iterations": 100,
            "learning_rate": 0.01,
            "use_cache": False,
            "description": "Test Settings"
        }
        colored_table, _ = show_in_table(**settings)
        self.assertIn("yes", colored_table)
        self.assertIn("no", colored_table)
        self.assertIn("\033[32m", colored_table)  # green for float
        self.assertIn("\033[31m", colored_table)  # red for boolean (False)
        self.assertIn("\033[35m", colored_table)  # magenta for int

    def test_titled_tabulate(self):
        data = {"abc": [1, 2, 3], "bcd": [4, 5, 6], 'cde': [7, 8, 9]}
        tit = titled_tabulate(
            "Test",
            '=',
            data,
            headers="keys",
            tablefmt=tabulate_formats.rounded_grid
        )

        lines = tit.split('\n')
        self.assertTrue('Test' in lines[1], f"Titled table is \n {tit}")
        self.assertEqual(len(tit[1]), len(tit[2]))

    def test_warn_unused_kwargs(self):
        empty1 = update_kwargs('test', {}, {'a': 1, 'b': 2})
        empty2 = update_kwargs('test', {}, {})
        res1 = update_kwargs('test', {'a': 1, 'b': 2}, {})
        res2 = update_kwargs('test', {'a': 1, 'b': 2}, {'a': 2})

        self.assertEqual(empty1, {})
        self.assertEqual(empty2, {})
        self.assertTrue(res1 == {'a': 1, 'b': 2})
        self.assertTrue(res2 == {'a': 2, 'b': 2})

    def test_terminate_popen_normal(self):
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdout = Mock()
        mock_process.stderr = Mock()
        mock_process.wait = Mock()

        terminate_popen(mock_process)

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

        terminate_popen(mock_process)

        mock_process.stdout.close.assert_called_once()
        mock_process.stderr.close.assert_called_once()
        mock_process.terminate.assert_called_once()
        self.assertEqual(mock_process.wait.call_count, 2)
        mock_process.wait.assert_any_call(timeout=5)
        mock_process.kill.assert_called_once()
