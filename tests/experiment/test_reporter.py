import time
from unittest.mock import patch

import matplotlib as plt

from pyfmto.experiment import Reports, RunSolutions
from pyfmto.experiment.reporter import Reporter, ReportGenerator
from tests.experiment import ExpTestCase

plt.use('Agg')


class ReporterTestBase(ExpTestCase):
    def gen_fake_data(self):
        for exp in self.conf.reporter.experiments:
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
        self.gen_fake_data()

        class FakeGenerator(ReportGenerator):
            data_size_req = 1

            def _generate(self, *args, **kwargs):
                pass
        reports = Reports(self.conf.reporter)
        reports.reporter.register_generator('fake', FakeGenerator())
        reports.to_curve()
        reports.to_curve(showing_size=10)
        reports.to_curve(suffix='.svg')
        reports.to_violin()
        reports.to_excel()
        reports.to_latex()
        reports.to_console()

        self.assertEqual(str(self.conf.launcher.results), str(self.conf.reporter.results))
        reports_dir = self.conf.reporter.root / time.strftime('%Y-%m-%d')
        self.assertTrue(reports_dir.exists())
        with self.assertRaises(ValueError):
            reports.reporter.generate_report('fake', [], '', '')


class TestReportsGenerate(ReporterTestBase):

    def test_generate_invalid_format(self):
        reports = Reports(self.conf.reporter)
        with self.assertRaises(ValueError):
            reports.reporter.generate_report('invalid', [], '', '')

        reports.conf.formats = []
        with self.assertRaises(ValueError):
            reports.generate()

        reports.conf.formats = ['invalid', 'curve']
        with self.assertRaises(ValueError):
            reports.generate()

        with self.assertRaises(ValueError):
            reports.reporter.generate_report('to_curve', ['ALGG'], 'PROB', 'NPD1')
        with patch('pyfmto.experiment.reporter.ReporterUtils.load_runs_data', return_value=[]):
            Reporter(self.conf.reporter)

    def test_generate_report_raises(self):
        reports = Reports(self.conf.reporter)

        class ReportWithError(Reporter):
            def generate_report(self, *args, **kwargs):
                raise ValueError("Test error")
        reports.reporter = ReportWithError(self.conf.reporter)
        reports.conf.formats = ['curve', 'excel', 'latex', 'console', 'violin']
        reports.generate()
