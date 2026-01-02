import unittest
import matplotlib as plt
from typing import Any
from unittest.mock import patch
from pathlib import Path

from pyfmto.experiments import RunSolutions, Reports
from pyfmto.experiments.reporter import ReportGenerator, Reporter
from pyfmto.utilities import save_yaml
from pyfmto.utilities.loaders import DataLoader
from tests.helpers import remove_temp_files, gen_problem, gen_algorithm

plt.use('Agg')


class ReporterTestBase(unittest.TestCase):
    def setUp(self):
        Path('out').mkdir(exist_ok=True)
        self.problems = ['PROB']
        self.algorithms = ['ALG1', 'ALG2']
        self.filename = 'out/config.yaml'
        self.conf_dict: dict[str, Any] = {
            'launcher': {
                'algorithms': self.algorithms,
                'problems': self.problems,
            },
            'problems': {self.problems[0]: {'dim': 20}},
        }
        save_yaml(self.conf_dict, self.filename)
        gen_algorithm(self.algorithms)
        gen_problem(self.problems)
        self.conf = DataLoader(self.filename).reporter
        self.gen_fake_data()
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
        reports = Reports(self.filename)
        reports.reporter.register_generator('fake', FakeGenerator())
        reports.to_curve()
        reports.to_curve(showing_size=10)
        reports.to_curve(suffix='.svg')
        reports.to_violin()
        reports.to_excel()
        reports.to_latex()
        reports.to_console()
        with self.assertRaises(ValueError):
            reports.reporter.generate_report('fake', [], '', '')


class TestReportsGenerate(ReporterTestBase):

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
            self.reports.reporter.generate_report('to_curve', ['ALGG'], 'PROB', 'NPD1')
        with patch('pyfmto.experiments.reporter.ReporterUtils.load_runs_data', return_value=[]):
            Reporter(self.conf)

    def test_generate_report_raises(self):
        class ReportWithError(Reporter):
            def generate_report(self, *args, **kwargs):
                raise ValueError("Test error")
        self.reports.reporter = ReportWithError(self.conf)
        self.reports.conf.formats = ['curve', 'excel', 'latex', 'console', 'violin']
        self.reports.generate()
