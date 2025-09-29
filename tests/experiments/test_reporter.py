import shutil
import unittest
import matplotlib as plt
from pathlib import Path
from pyfmto import load_problem, export_reporter_config
from pyfmto.experiments import RunSolutions, Reports
from pyfmto.experiments.reporter import (
    ReportGenerator, CurveGenerator, ViolinGenerator, ExcelGenerator, LatexGenerator, ConsoleGenerator, Reporter
)
from pyfmto.experiments.utils import MetaData
from tests.experiments import ExpDataGenerator

plt.use('Agg')


class TestGenerators(unittest.TestCase):
    def setUp(self):
        generator = ExpDataGenerator(dim=10, lb=-5, ub=5)
        self.data_ok = generator.gen_metadata(algs=['A1', 'A2'], prob='P1', npd='IID', n_tasks=5, n_runs=3)
        self.data_empty = MetaData({}, problem='TEST', npd_name='IID', filedir=Path('tmp/report'))

    def tearDown(self):
        shutil.rmtree(Path('tmp'), ignore_errors=True)

    def test_base_generator(self):
        class FakeGenerator(ReportGenerator):
            data_size_req = 1

            def _generate(self, *args, **kwargs):
                pass

        fg = FakeGenerator()
        fg.generate(self.data_ok)
        with self.assertRaises(ValueError):
            fg.generate(self.data_empty)

    def test_generators(self):
        curve = CurveGenerator()
        violin = ViolinGenerator()
        excel = ExcelGenerator()
        latex = LatexGenerator()
        consol = ConsoleGenerator()

        curve.generate(self.data_ok)
        curve.generate(self.data_ok, showing_size=10)
        curve.generate(self.data_ok, suffix='.svg')
        violin.generate(self.data_ok)
        excel.generate(self.data_ok)
        latex.generate(self.data_ok)
        consol.generate(self.data_ok)


class TestReporter(unittest.TestCase):
    def setUp(self):
        self.root = Path('out/results/')
        self.algs = ['ALG1', 'ALG2']
        self.probs = ['tetci2019', 'arxiv2017']
        self.gen_fake_data()
        export_reporter_config(algs=(self.algs, ), probs=tuple(self.probs), mode='update')
        self.reports = Reports()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        Path('config.yaml').unlink()

    def test_to_curve(self):
        self.reports.to_curve()
        self.reports.to_curve(showing_size=10)

    def test_to_excel(self):
        self.reports.to_excel()
        self.algs.append('ALG3')  # add a non-existing algorithm
        export_reporter_config(algs=(self.algs, ), probs=tuple(self.probs), mode='update')
        self.reports = Reports()
        self.reports.to_excel()

    def test_to_latex(self):
        self.reports.to_latex()

    def test_to_violin(self):
        self.reports.to_violin()

    def test_to_console(self):
        self.reports.to_console()

    def test_no_data(self):
        export_reporter_config(algs=(['ALG3', 'ALG4'], ), probs=tuple(self.probs), mode='update')
        rep = Reports()
        rep.to_curve()
        rep.to_latex()
        rep.to_violin()
        rep.to_console()
        rep.to_excel()

    def test_none_exist_generator(self):
        reporter = Reporter('', [])
        with self.assertRaises(ValueError):
            reporter.generate_report('non-exist', [], '', '')

    def gen_fake_data(self):
        for alg in self.algs:
            for prob in self.probs:
                res_root = self.root / alg / prob.upper() / 'IID'
                res_root.mkdir(parents=True, exist_ok=True)
                for run in range(5):
                    problem = load_problem(prob)
                    run_solutions = RunSolutions()
                    for task in problem:
                        x = task.random_uniform_x(task.fe_available)
                        y = task.evaluate(x)
                        task.solutions.append(x, y)
                        # print(f"task {task.id}: shapes(x,y) is [{task.solutions.x.shape}, {task.solutions.y.shape}]")
                        run_solutions.update(task.id, task.solutions)
                    run_solutions.to_msgpack(res_root / f"Run {run + 1}.msgpack")
