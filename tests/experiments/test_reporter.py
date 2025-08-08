import shutil
import unittest
import matplotlib as plt
from pathlib import Path

from pyfmto import load_problem, export_reporter_config
from pyfmto.experiments import RunSolutions, Reports

plt.use('Agg')


class TestValidReporting(unittest.TestCase):
    def setUp(self):
        self.root = Path('out/results/')
        self.algs = ['ALG1', 'ALG2']
        self.probs = ['tetci2019', 'arxiv2017']
        self.gen_fake_data()
        export_reporter_config(algs=[self.algs], probs=self.probs, mode='update')
        self.reports = Reports()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        Path('config.yaml').unlink()

    def test_reinit_data_hit_cache(self):
        self.reports.analyzer.init_data()

    def test_to_curve(self):
        self.reports.to_curve()
        self.reports.to_curve(showing_size=10)

    def test_to_excel(self):
        self.reports.to_excel()
        self.algs.append('ALG3')  # add a non-existing algorithm
        export_reporter_config(algs=[self.algs], probs=self.probs, mode='update')
        self.reports = Reports()
        self.reports.to_excel()

    def test_to_latex(self):
        self.reports.to_latex()

    def test_to_violin(self):
        self.reports.to_violin()

    def test_to_console(self):
        self.reports.to_console()

    def test_show_raw_results(self):
        self.reports.show_raw_results()

    def test_show_combinations(self):
        self.reports.show_combinations()

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
                        run_solutions.update(task.id, task.solutions)
                    run_solutions.to_msgpack(res_root / f"Run {run + 1}.msgpack")
