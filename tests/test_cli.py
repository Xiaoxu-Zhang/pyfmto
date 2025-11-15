import unittest
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
from pyfmto import main, update_path


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

    def test_update_path_with_algorithms_folder(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()

        with patch('pyfmto.Path.cwd', return_value=self.temp_dir):
            if str(self.temp_dir) in sys.path:
                sys.path.remove(str(self.temp_dir))
            update_path()
            self.assertIn(str(self.temp_dir), sys.path)

    def test_update_path_already_in_path(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()

        with patch('pyfmto.Path.cwd', return_value=self.temp_dir):
            if str(self.temp_dir) not in sys.path:
                sys.path.append(str(self.temp_dir))
            orig_len = len(sys.path)
            update_path()
            self.assertEqual(len(sys.path), orig_len)

    def test_update_path_without_algorithms_folder(self):
        with patch('pyfmto.Path.cwd', return_value=self.temp_dir):
            with self.assertRaises(FileNotFoundError):
                update_path()

    def test_update_path_current_directory_real(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()
        os.chdir(self.temp_dir)
        if str(self.temp_dir) in sys.path:
            sys.path.remove(str(self.temp_dir))
        update_path()
        self.assertIn(str(self.temp_dir), sys.path)

    def test_update_path_path_normalization(self):
        algorithms_dir = self.temp_dir / "algorithms"
        algorithms_dir.mkdir()

        with patch('pyfmto.Path.cwd', return_value=self.temp_dir):
            if str(self.temp_dir) in sys.path:
                sys.path.remove(str(self.temp_dir))
            update_path()
            self.assertIn(str(self.temp_dir), sys.path)
            path_index = sys.path.index(str(self.temp_dir))
            self.assertIsInstance(sys.path[path_index], str)


class TestMainFunction(unittest.TestCase):
    """Test cases for the main function in pyfmto.__init__"""

    def setUp(self):
        self.algorithms_dir = Path().cwd() / "algorithms"
        self.algorithms_dir.mkdir(parents=True, exist_ok=True)
        """Set up test fixtures before each test method."""
        self.original_argv = sys.argv

    def tearDown(self):
        """Clean up after each test method."""
        sys.argv = self.original_argv
        if self.algorithms_dir.exists():
            import shutil
            shutil.rmtree(self.algorithms_dir)

    @patch('pyfmto.Launcher')
    @patch('pyfmto.Reports')
    def test_run_command(self, mock_reports, mock_launcher):
        """Test that the main function correctly handles the 'run' command."""
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

    @patch('pyfmto.Launcher')
    @patch('pyfmto.Reports')
    def test_report_command(self, mock_reports, mock_launcher):
        """Test that the main function correctly handles the 'report' command."""
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

    @patch('pyfmto.Launcher')
    @patch('pyfmto.Reports')
    def test_default_config_file(self, mock_reports, mock_launcher):
        """Test that the main function uses default config when none is specified."""
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
