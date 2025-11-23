import unittest
from typing import Any

import matplotlib as plt
from pathlib import Path
from pyfmto.experiments import RunSolutions, Reports
from pyfmto.experiments.reporter import ReportGenerator, Reporter
from pyfmto.utilities import save_yaml
from pyfmto.utilities.loaders import ConfigLoader
from unittest.mock import patch
from tests.helpers import remove_temp_files, gen_problem, gen_algorithm

plt.use('Agg')


class ReporterTestBase(unittest.TestCase):
    def setUp(self):
        gen_algorithm(['ALG1', 'ALG2'])
        gen_problem('PROB1')
        Path('out').mkdir(exist_ok=True)
        self.filename = 'out/config.yaml'
        self.conf_dict: dict[str, Any] = {
            'launcher': {
                'algorithms': ['ALG1', 'ALG2', 'ALG3', 'ALG4'],
                'problems': ['PROB1'],
            },
            'problems': {'PROB1': {'dim': 20}},
        }
        save_yaml(self.conf_dict, self.filename)
        self.conf = ConfigLoader(self.filename).reporter
        self.reports = Reports(self.filename)

    def tearDown(self):
        remove_temp_files()

    def gen_fake_data(self):
        for exp in self.conf.experiments:
            exp.init_root()
            for run in range(5):
                prob = exp.problem.initialize()
                run_data = RunSolutions()
                for task in prob:
                    x = task.random_uniform_x(task.fe_available)
                    y = task.evaluate(x)
                    task.solutions.append(x, y)
                    run_data.update(task.id, task.solutions)
                run_data.to_msgpack(exp.result_name(run))


class TestGenerators(ReporterTestBase):

    def test_base_generator(self):
        class FakeGenerator(ReportGenerator):
            data_size_req = 1

            def _generate(self, *args, **kwargs):
                pass
        self.gen_fake_data()
        reports = Reports(self.filename)
        reports.reporter.register_generator('fake', FakeGenerator())
        with self.assertRaises(ValueError):
            reports.reporter.generate_report('fake', [], '', '')
        reports.to_curve()
        reports.to_curve(showing_size=10)
        reports.to_curve(suffix='.svg')
        reports.to_violin()
        reports.to_excel()
        reports.to_latex()
        reports.to_console()


class TestReportsGenerate(ReporterTestBase):

    @patch.object(Reports, 'to_curve')
    @patch.object(Reports, 'to_excel')
    @patch.object(Reports, 'to_latex')
    @patch.object(Reports, 'to_console')
    @patch.object(Reports, 'to_violin')
    def test_generate_valid_formats(self, mock_violin, mock_console, mock_latex, mock_excel, mock_curve):
        # Set up mock returns
        mock_curve.return_value = None
        mock_excel.return_value = None
        mock_latex.return_value = None
        mock_console.return_value = None
        mock_violin.return_value = None

        # Test with single format
        self.reports.conf.formats = ['curve']
        self.reports.generate()
        mock_curve.assert_called_once()
        mock_excel.assert_not_called()
        mock_latex.assert_not_called()
        mock_console.assert_not_called()
        mock_violin.assert_not_called()

        # Reset mocks
        mock_curve.reset_mock()
        mock_excel.reset_mock()
        mock_latex.reset_mock()
        mock_console.reset_mock()
        mock_violin.reset_mock()

        # Test with multiple formats
        self.reports.conf.formats = ['curve', 'excel', 'latex']
        self.reports.generate()
        mock_curve.assert_called_once()
        mock_excel.assert_called_once()
        mock_latex.assert_called_once()
        mock_console.assert_not_called()
        mock_violin.assert_not_called()

    def test_generate_invalid_format(self):
        with self.assertRaises(ValueError):
            self.reports.reporter.generate_report('invalid', [], '', '')

    def test_generate_empty_formats(self):
        self.reports.conf.formats = []
        with self.assertRaises(ValueError):
            self.reports.generate()

    def test_invalid_format_in_formats(self):
        self.reports.conf.formats = ['invalid', 'curve']
        with self.assertRaises(ValueError):
            self.reports.generate()

    def test_no_existing_data(self):
        with self.assertRaises(ValueError):
            self.reports.reporter.generate_report('to_curve', ['ALGG'], 'PROB1', 'NPD1')

    def test_generate_report_raises(self):
        class ReportWithError(Reporter):
            def generate_report(self, *args, **kwargs):
                raise ValueError("Test error")
        self.reports.reporter = ReportWithError(self.conf)
        self.reports.conf.formats = ['curve', 'excel', 'latex', 'console', 'violin']
        self.reports.generate()
