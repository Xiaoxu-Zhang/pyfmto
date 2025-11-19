import unittest
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from pyfmto import list_problems, list_algorithms
from pyfmto.experiments import list_report_formats
from pyfmto.utilities.cli import update_path, main
from tests.experiments import export_alg_template


class TestUpdatePath(unittest.TestCase):
    def setUp(self):
        self.original_sys_path = sys.path.copy()
        self.original_cwd = os.getcwd()
        self.temp_dir = Path().cwd() / "tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        sys.path[:] = self.original_sys_path
        os.chdir(self.original_cwd)
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_update_path_already_in_path(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()

        with patch('pyfmto.utilities.cli.Path.cwd', return_value=self.temp_dir):
            if str(self.temp_dir) not in sys.path:
                sys.path.append(str(self.temp_dir))
            orig_len = len(sys.path)
            update_path()
            self.assertEqual(len(sys.path), orig_len)

    def test_update_path_path_normalization(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()

        with patch('pyfmto.utilities.cli.Path.cwd', return_value=self.temp_dir):
            if str(self.temp_dir) in sys.path:
                sys.path.remove(str(self.temp_dir))
            update_path()
            self.assertIn(str(self.temp_dir), sys.path)
            path_index = sys.path.index(str(self.temp_dir))
            self.assertIsInstance(sys.path[path_index], str)


class TestMainFunction(unittest.TestCase):

    def setUp(self):
        self.algorithms_dir = Path().cwd() / "algorithms"
        self.algorithms_dir.mkdir(parents=True, exist_ok=True)
        self.original_argv = sys.argv

    def tearDown(self):
        sys.argv = self.original_argv
        if self.algorithms_dir.exists():
            import shutil
            shutil.rmtree(self.algorithms_dir)

    @patch('pyfmto.utilities.cli.Launcher')
    @patch('pyfmto.utilities.cli.Reports')
    def test_run_command(self, mock_reports, mock_launcher):
        # Setup mock launcher
        mock_launcher_instance = Mock()
        mock_launcher.return_value = mock_launcher_instance

        # Set up command line arguments
        test_args = ['pyfmto', 'run', '--config', 'test_config.yaml']
        with patch.object(sys, 'argv', test_args):
            main()

        # Verify launcher was created with correct config and run was called
        mock_launcher.assert_called_once_with(conf_file='test_config.yaml')
        mock_launcher_instance.run.assert_called_once()
        mock_reports.assert_not_called()

    @patch('pyfmto.utilities.cli.Launcher')
    @patch('pyfmto.utilities.cli.Reports')
    def test_report_command(self, mock_reports, mock_launcher):
        # Setup mock reports
        mock_reports_instance = Mock()
        mock_reports.return_value = mock_reports_instance

        # Set up command line arguments
        test_args = ['pyfmto', 'report', '--config', 'test_config.yaml']
        with patch.object(sys, 'argv', test_args):
            main()

        # Verify reports was created with correct config and generate was called
        mock_reports.assert_called_once_with(conf_file='test_config.yaml')
        mock_reports_instance.generate.assert_called_once()
        mock_launcher.assert_not_called()

    @patch('pyfmto.utilities.cli.Launcher')
    @patch('pyfmto.utilities.cli.Reports')
    def test_default_config_file(self, mock_reports, mock_launcher):
        # Setup mock launcher
        mock_launcher_instance = Mock()
        mock_launcher.return_value = mock_launcher_instance

        # Set up command line arguments without config
        test_args = ['pyfmto', 'run']
        with patch.object(sys, 'argv', test_args):
            main()

        # Verify launcher was created with default config
        mock_launcher.assert_called_once_with(conf_file='config.yaml')
        mock_launcher_instance.run.assert_called_once()

    def test_list_command(self):
        options = ['algorithms', 'problems', 'reports']
        args_lst = [['pyfmto', 'list', option] for option in options]
        for test_args in args_lst:
            with self.subTest(test_args=test_args):
                with patch.object(sys, 'argv', test_args):
                    main()

    def test_show_command(self):
        export_alg_template('ALG1')
        export_alg_template('ALG2')
        options = {
            'prob': list(list_problems().keys()),
            'report': list_report_formats(),
            'alg': list(list_algorithms().keys()),
            'invalid': ['Invalid']
        }
        for grp, lst in options.items():
            for option in lst:
                with self.subTest(grp=grp, option=option):
                    test_args = ['pyfmto', 'show', f"{grp}.{option}"]
                    with patch.object(sys, 'argv', test_args):
                        main()
