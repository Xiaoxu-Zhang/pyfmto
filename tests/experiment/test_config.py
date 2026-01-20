import os
from pathlib import Path
from unittest.mock import patch

import pyfmto.experiment
from pyfmto.experiment.config import Config, ConfigLoader, LauncherConfig, ReporterConfig
from pyfmto.framework import AlgorithmData
from pyfmto.problem import ProblemData
from pyfmto.utilities.io import dumps_yaml
from tests.helpers import PyfmtoTestCase
from tests.helpers.testcases import TestCaseAlgProbConf


class TestOthers(PyfmtoTestCase):
    def test_combine_params(self):
        params = {
            'a': ['a'],
            'b': ['b1', 'b2'],
            'c': [True, False],
            'd': 'd'
        }
        expected = [
            {'a': 'a', 'b': 'b1', 'c': True, 'd': 'd'},
            {'a': 'a', 'b': 'b2', 'c': True, 'd': 'd'},
            {'a': 'a', 'b': 'b1', 'c': False, 'd': 'd'},
            {'a': 'a', 'b': 'b2', 'c': False, 'd': 'd'},
        ]
        res = pyfmto.experiment.config.combine_params(params)
        for r in res:
            with self.subTest(r=r):
                self.assertIn(r, expected, msg=f"\n  result: {r} not in\nexpected: {expected}")


class TestConfigBase(PyfmtoTestCase):
    def test_config_base(self):
        config = Config(
            results='out/results',
        )
        config.algorithms = ['a1', 'a2', 'a3']
        config.problems = ['p1', 'p2', 'p3']
        config.algorithms_data = [AlgorithmData(), AlgorithmData(), AlgorithmData()]
        config.problems_data = [ProblemData(), ProblemData(), ProblemData()]
        self.assertEqual(config.algorithms, ['a1', 'a2', 'a3'])
        self.assertEqual(config.problems, ['p1', 'p2', 'p3'])
        self.assertEqual(config.n_exp, len(list(config.experiments)))


class TestExperimentData(TestCaseAlgProbConf):

    def test_default_value(self):
        alg = self.load_algorithm(self.alg_names[0])
        prob = self.load_problem(self.prob_names[0])
        self.assertTrue(alg.available, msg=self.list_algorithms())
        self.assertTrue(prob.available, msg=self.list_problems())
        exp = pyfmto.experiment.config.ExperimentData(alg, prob, 'out/results')
        self.assertIn('Alg', repr(exp))
        self.assertIn('Prob', repr(exp))
        self.assertIsInstance(exp.algorithm, AlgorithmData)
        self.assertIsInstance(exp.problem, ProblemData)
        self.assertIsInstance(exp.result_dir, Path)
        self.assertEqual(
            exp.result_dir,
            Path('out/results') / exp.algorithm.name / exp.problem.name / exp.problem.npd_str
        )
        self.assertIn("FEi", exp.prefix)
        self.assertIn('FEm', exp.prefix)
        self.assertIn('Seed', exp.prefix)
        self.assertIn('Rep01.msgpack', exp.result_name(1).name)

        self.assertEqual(exp.n_results, 0)
        exp.init_root()
        for i in range(5):
            with open(exp.result_name(i), 'w') as f:
                f.write('---')
        self.assertEqual(exp.n_results, 5)

        self.assertIn('algorithm', exp.params_snapshot)
        self.assertIn('problem', exp.params_snapshot)
        self.assertIn('PROB', str(exp))
        self.assertIn('ALG', str(exp))

        exp.problem.params_update = {'dim': 2, 'npd': 2}
        prefix = "FEi10_FEm22_Seed123_"
        self.assertEqual(exp.prefix, prefix)
        exp.problem.params_update = {'dim': 2, 'npd': 2, 'fe_init': 6, 'fe_max': 12}
        prefix = "FEi6_FEm12_Seed123_"
        self.assertEqual(exp.prefix, prefix, msg=dumps_yaml(exp.problem._merged_params))

        self.assertEqual(exp.n_results, 0)
        exp.init_root()
        for i in range(3):
            with open(exp.result_name(i), 'w') as f:
                f.write('---')
        self.assertEqual(exp.n_results, 3)
        res_root = exp.result_dir.parent.parent
        n_prob = len(os.listdir(res_root))
        self.assertEqual(n_prob, 1, msg=f"res root: {res_root}")

        self.assertEqual(exp.problem.dim, 2)
        self.assertEqual(exp.problem.npd, 2)
        self.assertEqual(exp.problem.dim_str, '2D')
        self.assertEqual(exp.problem.npd_str, 'NPD2')
        self.assertEqual(exp.problem.params['fe_init'], 6)
        self.assertEqual(exp.problem.params['fe_max'], 12)

        exp.create_snapshot([])
        exp.create_snapshot([])  # cover the snapshot exists case
        self.assertTrue(exp.result_dir.parent.exists())
        self.assertTrue(exp.code_dest.exists())
        self.assertGreater(len(list(exp.code_dest.iterdir())), 0)
        self.assertTrue(exp.markdown_dest.exists())


class TestConfigLoader(PyfmtoTestCase):

    def setUp(self):
        super().setUp()

        self.valid = f"""
        launcher:
            sources: [{self.tmp_dir}]
            algorithms: [ALG1, ALG2]
            problems: [PROB1, PROB2]
        """
        self.invalid = """
        launcher:
            results: ''
            repeat: 0
            save: yes
            loglevel: INFO
            algorithms: []
            problems: []
        reporter:
            results: ''
            comparisons: []
            problems: []
            formats: []
        """

    def test_default_config(self):
        filename = self.gen_config(self.valid)
        conf = ConfigLoader(filename)
        self.assertIn('launcher', conf.config)
        self.assertIn('reporter', conf.config)
        self.assertIn('results', conf.config['reporter'])
        self.assertIn('comparisons', conf.config['reporter'])
        self.assertIn('problems', conf.config['reporter'])
        self.assertIn('formats', conf.config['reporter'])
        l_conf = conf.config['launcher']
        r_conf = conf.config['reporter']
        self.assertEqual(l_conf['results'], r_conf['results'])
        self.assertEqual(l_conf['problems'], r_conf['problems'])
        self.assertEqual(l_conf['algorithms'], ['ALG1', 'ALG2'])
        self.assertEqual(r_conf['comparisons'], [['ALG1', 'ALG2']])
        self.assertEqual(len(conf.config), 2)
        self.gen_config(
            """
            launcher:
                results: out/launcher/results
                algorithms: [ALG1, ALG2]
                problems: [PROB1, PROB2]
            reporter:
                results: out/reporter/results
            problems:
                PROB1: []
            """
        )
        conf = ConfigLoader(filename)
        self.assertEqual(len(conf.config), 3)
        self.assertIn('launcher', conf.config)
        self.assertIn('reporter', conf.config)
        self.assertIn('problems', conf.config)
        self.assertEqual(conf.config['launcher']['results'], 'out/launcher/results')
        self.assertEqual(conf.config['reporter']['results'], 'out/reporter/results')
        self.assertEqual(conf.config['launcher']['algorithms'], ['ALG1', 'ALG2'])
        self.assertEqual(conf.config['reporter']['comparisons'], [['ALG1', 'ALG2']])
        self.assertEqual(conf.config['launcher']['problems'], ['PROB1', 'PROB2'])
        self.assertEqual(conf.config['reporter']['problems'], ['PROB1', 'PROB2'])

    def test_check_config(self):
        self.gen_algs(['ALG1', 'ALG2'])
        self.gen_probs(['PROB1', 'PROB2'])
        filename = self.gen_config(self.invalid)
        conf = ConfigLoader(filename)
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')

        self.gen_config(
            """
            launcher:
                sources: []
                results: ''
                repeat: 0
                save: yes
                loglevel: INFO
                algorithms: []
                problems: []
            reporter:
                results: ''
                comparisons: [123, ALG1, ALG2, [ALG3, ALG4]]
                problems: []
                formats: []
            """
        )
        conf = ConfigLoader(filename)
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')

    def test_conf_values(self):
        filename = self.gen_config(self.valid)
        conf = ConfigLoader(filename)
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 0)
        self.assertEqual(list(conf.reporter.experiments), [])

    def test_component_exists(self):
        filename = self.gen_config(self.valid)
        self.gen_algs(['ALG1', 'ALG2'])
        self.gen_probs(['PROB1', 'PROB2'])
        conf = ConfigLoader(filename)
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 4)
        self.assertEqual(len(list(conf.reporter.experiments)), 4)

    @patch('pyfmto.experiment.config.discover')
    def test_component_not_available(self, mock_discover):
        mock_discover.return_value = {
            'algorithms': {
                'ALG1': [AlgorithmData()],
                'ALG2': [AlgorithmData()],
            },
            'problems': {
                'PROB1': [ProblemData()],
                'PROB2': [ProblemData()],
            }
        }
        filename = self.gen_config(self.valid)
        conf = ConfigLoader(filename)
        self.assertEqual(conf.launcher.algorithms_data, [])
        self.assertEqual(conf.launcher.problems_data, [])
