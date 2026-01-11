import subprocess
import warnings
from unittest.mock import patch, Mock, MagicMock, mock_open

from pyfmto.utilities import (
    colored,
    clear_console,
    titled_tabulate,
    terminate_popen,
    tabulate_formats,
)
from pyfmto.utilities.tools import redirect_warnings, print_dict_as_table, get_pkgs_version, get_cpu_model, get_os_name

from tests.helpers import PyfmtoTestCase


class TestTools(PyfmtoTestCase):

    def tearDown(self):
        self.delete()

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

    def test_dict_to_table(self):
        data_valid = {
            "a": [1, 2, 3],
            "b": [True, False, False],
            "c": ['OK', 'Yes', 'NO'],
        }
        data_invalid = {
            "a": [1, 2, 3],
            "b": [True, False],
        }
        print_dict_as_table(data_valid)
        with self.assertRaises(ValueError):
            print_dict_as_table(data_invalid)
        print_dict_as_table({})

    def test_colored(self):
        text = "test"
        color_text = colored(text, 'green')
        self.assertIn('\033[32m', color_text)
        self.assertIn(text, color_text)
        self.assertRaises(ValueError, colored, text, 'invalid_color')

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

    def test_redirect_warnings_user_warning(self):
        with redirect_warnings():
            with patch('pyfmto.utilities.loggers.logger.warning') as mock_logger_warning:
                warnings.warn("This is a user warning", UserWarning)
                mock_logger_warning.assert_called_once()

    def test_redirect_warnings_deprecation_warning(self):
        mock_original = Mock()
        warnings.showwarning = mock_original
        with redirect_warnings():
            warnings.warn("This is a deprecation warning", DeprecationWarning)
            mock_original.assert_called_once()


class TestGetOsName(PyfmtoTestCase):

    @patch('sys.platform', 'win32')
    def test_windows(self):
        self.assertEqual(get_os_name(), 'Windows')

    @patch('sys.platform', 'darwin')
    def test_macos(self):
        self.assertEqual(get_os_name(), 'macOS')

    @patch('sys.platform', 'linux')
    def test_linux(self):
        self.assertEqual(get_os_name(), 'Linux')

    @patch('sys.platform', 'freebsd13')
    def test_other_unix_like(self):
        # Any non-win32/darwin platform returns 'Linux' per current logic
        self.assertEqual(get_os_name(), 'Linux')


class TestGetPkgsVersion(PyfmtoTestCase):

    def test_importlib_metadata_success(self):
        with patch('importlib.metadata.version') as mock_version:
            mock_version.return_value = '3.10.0'
            result = get_pkgs_version(['requests'])
            self.assertEqual(result, {'requests': '3.10.0'})

    def test_importlib_fails___version___success(self):
        with patch('importlib.metadata.version', side_effect=Exception("Not found")):
            with patch('importlib.import_module') as mock_import:
                mock_mod = MagicMock()
                mock_mod.__version__ = '1.9.3'
                mock_import.return_value = mock_mod

                result = get_pkgs_version(['flask'])
                self.assertEqual(result, {'flask': '1.9.3'})

    def test_both_methods_fail(self):
        with patch('importlib.metadata.version', side_effect=Exception("metadata fail")):
            with patch('importlib.import_module', side_effect=ImportError("no such module")):
                result = get_pkgs_version(['nonexistent_pkg'])
                self.assertEqual(result, {'nonexistent_pkg': 'unknown'})

    def test_mixed_packages(self):

        def fake_version(name):
            if name == 'pkg_a':
                return '2.0'
            else:
                raise Exception("not found")

        with patch('importlib.metadata.version', side_effect=fake_version):
            with patch('importlib.import_module') as mock_import:
                def fake_import(name):
                    if name == 'pkg_b':
                        mod = MagicMock()
                        mod.__version__ = '5.1'
                        return mod
                    elif name == 'pkg_c':
                        raise ImportError()
                    else:
                        raise Exception("unexpected")
                mock_import.side_effect = fake_import

                result = get_pkgs_version(['pkg_a', 'pkg_b', 'pkg_c'])
                expected = {
                    'pkg_a': '2.0',
                    'pkg_b': '5.1',
                    'pkg_c': 'unknown'
                }
                self.assertEqual(result, expected)


class TestGetCpuModel(PyfmtoTestCase):

    @patch('platform.processor')
    def test_processor_returns_valid_cpu(self, mock_proc):
        mock_proc.return_value = 'Intel(R) Core(TM) i7-12700K'
        self.assertEqual(get_cpu_model(), 'Intel(R) Core(TM) i7-12700K')

    @patch('platform.processor')
    def test_processor_returns_generic_x86_64(self, mock_proc):
        mock_proc.return_value = 'x86_64'
        # Should skip and try OS-specific methods
        with patch('sys.platform', 'darwin'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(stdout='Apple M3 Pro\n')
                self.assertEqual(get_cpu_model(), 'Apple M3 Pro')

    @patch('platform.processor')
    def test_macos_success(self, mock_proc):
        mock_proc.return_value = ''  # triggers fallback
        with patch('sys.platform', 'darwin'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(stdout='Intel(R) Xeon CPU E5-2690 v4 @ 2.60GHz')
                self.assertEqual(get_cpu_model(), 'Intel(R) Xeon CPU E5-2690 v4 @ 2.60GHz')

    @patch('platform.processor')
    def test_macos_subprocess_fails(self, mock_proc):
        mock_proc.return_value = ''
        with patch('sys.platform', 'darwin'):
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'sysctl')):
                # No other fallback on macOS → final fallback
                self.assertEqual(get_cpu_model(), 'Unknown CPU')

    @patch('platform.processor')
    def test_linux_success(self, mock_proc):
        mock_proc.return_value = 'arm'  # generic, so fallback to /proc/cpuinfo
        cpuinfo_content = """
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model name      : AMD EPYC 7763 64-Core Processor
"""
        with patch('sys.platform', 'linux'):
            with patch('builtins.open', mock_open(read_data=cpuinfo_content)):
                self.assertEqual(get_cpu_model(), 'AMD EPYC 7763 64-Core Processor')

    @patch('platform.processor')
    def test_linux_no_model_name_in_cpuinfo(self, mock_proc):
        mock_proc.return_value = ''
        with patch('sys.platform', 'linux'):
            with patch('builtins.open', mock_open(read_data="cpu MHz: 3000.000\n")):
                self.assertEqual(get_cpu_model(), 'Unknown CPU')

    @patch('platform.processor')
    def test_linux_file_access_error(self, mock_proc):
        mock_proc.return_value = ''
        with patch('sys.platform', 'linux'):
            with patch('builtins.open', side_effect=OSError("Permission denied")):
                self.assertEqual(get_cpu_model(), 'Unknown CPU')

    @patch('platform.processor')
    def test_windows_or_other_platform(self, mock_proc):
        mock_proc.return_value = 'x86_64'
        with patch('sys.platform', 'win32'):
            # Doesn't enter macOS/Linux branches
            self.assertEqual(get_cpu_model(), 'Unknown CPU')
