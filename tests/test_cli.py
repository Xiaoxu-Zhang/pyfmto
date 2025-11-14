import unittest
from unittest.mock import patch, Mock
import sys

from pyfmto import main


class TestMainFunction(unittest.TestCase):
    """Test cases for the main function in pyfmto.__init__"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.original_argv = sys.argv

    def tearDown(self):
        """Clean up after each test method."""
        sys.argv = self.original_argv

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
