import time
from pathlib import Path
from unittest.mock import patch

import matplotlib as plt

from pyfmto.experiment import Reporter, RunSolutions
from pyfmto.experiment.reporter import GeneratorManager, ReportGenerator
from tests.helpers.testcases import TestCaseAlgProbConf

plt.use('Agg')


class ReporterTestBase(TestCaseAlgProbConf):
    def gen_fake_data(self):
        for exp in self.config.reporter.experiments:
            exp.init_root()
            print(f"initialized data root: {exp.result_dir}")
            for run in range(5):
                prob = exp.problem.initialize()
                run_data = RunSolutions()
                for task in prob:
                    x = task.random_uniform_x(task.fe_available)
                    y = task.evaluate(x)
                    task.solutions.append(x, y)
                    run_data.update(task.id, task.solutions)
                filename = exp.result_name(run)
                run_data.to_msgpack(filename)
                print(f"Saved: {filename}")


class TestGenerators(ReporterTestBase):

    def test_base_generator(self):
        self.gen_fake_data()

        class FakeGenerator(ReportGenerator):
            data_size_req = 1

            def _generate(self, *args, **kwargs):
                pass
        reporter = Reporter(self.config.reporter)
        reporter.manager.register_generator('fake', FakeGenerator())
        reporter.to_curve()
        reporter.to_curve(showing_size=10)
        reporter.to_curve(suffix='.svg')
        reporter.to_violin()
        reporter.to_excel()
        reporter.to_latex()
        reporter.to_console()

        self.assertEqual(self.config.launcher.results, self.config.reporter.results)
        reports_dir = Path(self.config.reporter.results) / time.strftime('%Y-%m-%d')
        self.assertTrue(reports_dir.exists())
        with self.assertRaises(ValueError):
            reporter.manager.generate_report('fake', [], '', '')


class TestReportsGenerate(ReporterTestBase):

    def test_generate_invalid_format(self):
        reporter = Reporter(self.config.reporter)
        with self.assertRaises(ValueError):
            reporter.manager.generate_report('invalid', [], '', '')

        reporter.conf.formats = []
        with self.assertRaises(ValueError):
            reporter.report()

        reporter.conf.formats = ['invalid', 'curve']
        with self.assertRaises(ValueError):
            reporter.report()

        with self.assertRaises(ValueError):
            reporter.manager.generate_report('to_curve', ['ALGG'], 'PROB', 'NPD1')
        with patch('pyfmto.experiment.reporter.ReporterUtils.load_runs_data', return_value=[]):
            GeneratorManager(self.config.reporter)

    def test_generate_report_raises(self):
        reporter = Reporter(self.config.reporter)

        class ReportWithError(GeneratorManager):
            def generate_report(self, *args, **kwargs):
                raise ValueError("Test error")
        reporter.manager = ReportWithError(self.config.reporter)
        reporter.conf.formats = ['curve', 'excel', 'latex', 'console', 'violin']
        reporter.report()
