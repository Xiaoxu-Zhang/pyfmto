import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from pyfmto.experiment import list_report_formats
from pyfmto.utilities.cli import main
from tests.helpers import gen_code, PyfmtoTestCase
from tests.helpers.generators import gen_config


class TestMainFunction(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        self.tmp_file = 'temp_conf.yaml'

    def tearDown(self):
        self.restore_sys_env()

    @patch('pyfmto.utilities.cli.Launcher')
    @patch('pyfmto.utilities.cli.ConfigLoader')
    def test_run_command(self, mock_loader, mock_launcher):
        mock_launcher_instance = Mock()
        mock_launcher.return_value = mock_launcher_instance
        mock_loader.return_value = Mock()

        test_args = ['pyfmto', 'run', '--config', self.tmp_file]
        with patch.object(sys, 'argv', test_args):
            main()

        mock_loader.assert_called_once_with(config=self.tmp_file)
        mock_launcher.assert_called_once()
        mock_launcher_instance.run.assert_called_once()

    @patch('pyfmto.utilities.cli.Reports')
    @patch('pyfmto.utilities.cli.ConfigLoader')
    def test_report_command(self, mock_loader, mock_reports):
        mock_reports_instance = Mock()
        mock_reports.return_value = mock_reports_instance
        mock_loader.return_value = Mock()

        test_args = ['pyfmto', 'report', '--config', self.tmp_file]
        with patch.object(sys, 'argv', test_args):
            main()

        mock_loader.assert_called_once_with(config=self.tmp_file)
        mock_reports.assert_called_once()
        mock_reports_instance.generate.assert_called_once()

    @patch('pyfmto.utilities.cli.ConfigLoader')
    def test_default_config_file(self, mock_loader):
        test_args = ['pyfmto', 'run']
        with patch.object(sys, 'argv', test_args):
            main()
        mock_loader.assert_called_once_with(config='config.yaml')

    def test_show_command(self):
        tmp_dir = Path('temp_dir_for_test')
        algs = ['ALG1', 'ALG2']
        probs = ['PROB1', 'PROB2']
        gen_code('algorithms', algs, tmp_dir)
        gen_code('problems', probs, tmp_dir)
        conf_file = gen_config(
            f"""
            launcher:
                sources: [{tmp_dir}]
                algorithms: [{algs[0]}, {algs[1]}]
                problems: [{probs[0]}, {probs[1]}]
            """,
            tmp_dir
        )
        show_options = {
            'prob': algs,
            'report': list_report_formats(),
            'alg': probs,
            'invalid': ['Invalid']
        }
        for grp, lst in show_options.items():
            for option in lst:
                with self.subTest(grp=grp, option=option):
                    test_args = ['pyfmto', 'show', f'{grp}.{option}', '-c', f'{conf_file}']
                    with patch.object(sys, 'argv', test_args):
                        main()

        self.restore_sys_env()
        self.save_sys_env()
        list_options = ['algorithms', 'problems', 'reports']
        args_lst = [['pyfmto', 'list', f'{option}', '-c', f'{conf_file}'] for option in list_options]
        for test_args in args_lst:
            with self.subTest(test_args=test_args):
                with patch.object(sys, 'argv', test_args):
                    main()
        self.delete(tmp_dir)
        self.restore_sys_env()
