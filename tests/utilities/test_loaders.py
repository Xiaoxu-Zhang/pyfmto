import os
import unittest
from pathlib import Path
from pydantic import ValidationError
from ruamel.yaml import CommentedMap

from pyfmto.problem import MultiTaskProblem
from pyfmto.utilities import loaders
from pyfmto.experiment import list_report_formats, show_default_conf
from pyfmto.utilities.loaders import (
    ProblemData, AlgorithmData,
    LauncherConfig, ReporterConfig,
    ConfigLoader
)
from tests.helpers import gen_code, PyfmtoTestCase
from tests.helpers.generators import gen_config
from tests.utilities import LoadersTestCase


class TestUtilities(PyfmtoTestCase):

    def test_recursive_to_pure_dict(self):
        data = {
            'a': {'m': 1},
            'b': {'n': CommentedMap(e=2)},
            'c': [1, 2]
        }
        expected = {
            'a': {'m': 1},
            'b': {'n': {'e': 2}},
            'c': [1, 2]
        }
        res = loaders.recursive_to_pure_dict(data)
        self.assertIsInstance(res, dict)
        self.assertIsInstance(res['a'], dict)
        self.assertIsInstance(res['b'], dict)
        self.assertIsInstance(res['c'], list)
        self.assertIsInstance(res['b']['n'], dict)
        self.assertEqual(res, expected, msg=f"\n  result: {res}\nexpected: {expected}")

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
        res = loaders.combine_params(params)
        for r in res:
            with self.subTest(r=r):
                self.assertIn(r, expected, msg=f"\n  result: {r} not in\nexpected: {expected}")

    def test_list_report_formats(self):
        res = list_report_formats(print_it=True)
        for fmt in ['curve', 'violin', 'excel', 'latex', 'console']:
            self.assertIn(fmt, res)

    def test_show_default_conf(self):
        for fmt in list_report_formats() + ['nonexist']:
            with self.subTest(fmt=fmt):
                self.assertIsNone(show_default_conf(fmt))

    def test_load_problem(self):
        self.save_sys_env()

        tmp_dir = Path('temp_dir_for_test')
        conf_file = gen_config(
            f"""
            launcher:
                sources: [{tmp_dir}]
            """,
            tmp_dir
        )
        self.probs = ['PROB1']
        gen_code('problems', self.probs, tmp_dir)
        res = loaders.load_problem(self.probs[0], conf_file)
        self.assertIsInstance(res, MultiTaskProblem)
        with self.assertRaises(ValueError):
            loaders.load_problem('PROB2', conf_file)

        self.delete(tmp_dir)
        self.restore_sys_env()


class TestAlgorithmData(LoadersTestCase):

    def test_default_value(self):
        alg = self.algorithms.get('ALG1')
        self.assertEqual(alg.params_default, {'client': {'name': 'c'}, 'server': {'name': 's'}})
        self.assertEqual(alg.params_default, alg.params)
        self.assertEqual(alg.name, 'ALG1')
        self.assertEqual(alg.name_alias, '')
        self.assertTrue('client' in alg.params_yaml)

    def test_update_params(self):
        alg = self.algorithms.get('ALG1')
        para_dft = {'client': {'name': 'c'}}
        para_upd = {'client': {'name': 'cc', 'c_new': 'n'}, 'server': {'name': 'ss', 's_new': 'n'}}
        alg.params_default = para_dft
        self.assertEqual(alg.params, para_dft)
        self.assertEqual(alg.params_diff, '')
        alg.params_update = para_upd
        self.assertEqual(alg.params, alg.params_update)
        self.assertTrue(alg.params_diff != '')
        alg.params_default = {}
        alg.params_update = {}
        self.assertIn('no configurable parameters', alg.params_yaml)
        alg.name_alias = 'ALGG'
        self.assertEqual(alg.name_alias, 'ALGG')
        self.assertEqual(alg.name, 'ALGG')

    def test_load_failure(self):
        empty_alg = self.tmp_dir / 'algorithms' / 'ALG2'
        empty_alg.mkdir(parents=True, exist_ok=True)
        module_path = '.'.join(empty_alg.parts[-3:])
        alg = AlgorithmData('ALG2', [module_path])
        msg = '\n'.join(alg.verbose()['msg']) + module_path
        self.assertIn("The subclass of 'Client' not found.", msg, msg=msg)
        self.assertIn("The subclass of 'Server' not found.", msg, msg=msg)
        self.assertFalse(hasattr(alg, 'client'))
        self.assertFalse(hasattr(alg, 'server'))
        self.assertFalse(alg.available)

        alg = AlgorithmData('ALG2', ["invalid_path.algorithms.ALG2"])
        msg = '\n'.join(alg.verbose()['msg'])
        self.assertIn('No module named', msg, msg=msg)

    def test_copy(self):
        alg = self.algorithms.get('ALG1')
        alg_cp = alg.copy()
        self.assertTrue(id(alg) != id(alg_cp))
        self.assertEqual(alg.__dict__, alg_cp.__dict__)


class TestProblemData(LoadersTestCase):

    def test_default_value(self):
        prob = self.problems.get('PROB1')
        prob.params_default.pop('dim')
        self.assertIn('PROB1', prob.name)
        self.assertEqual(prob.npd, 1)
        self.assertEqual(prob.npd_str, 'NPD1')
        self.assertEqual(prob.params_diff, '')
        self.assertEqual(prob.dim, 0)
        self.assertEqual(prob.dim_str, '')
        self.assertNotIn('fe_init', prob.params)
        self.assertNotIn('fe_max', prob.params)

        prob.params_default.update({'dim': 1})
        self.assertEqual(prob.dim, 1)
        self.assertEqual(prob.dim_str, '1D')
        self.assertNotEqual(prob.params_diff, '')
        self.assertEqual(prob.params['fe_init'], 5 * prob.dim)
        self.assertEqual(prob.params['fe_max'], 11 * prob.dim)

        prob.params_update = {'dim': 2, 'npd': 2}
        self.assertEqual(prob.dim, 2)
        self.assertEqual(prob.npd, 2)
        self.assertEqual(prob.dim_str, '2D')
        self.assertEqual(prob.npd_str, 'NPD2')
        self.assertEqual(prob.params['fe_init'], 5 * prob.dim)
        self.assertEqual(prob.params['fe_max'], 11 * prob.dim)
        self.assertTrue(prob.params_diff != '')

        self.assertNotIn('no configurable parameters', prob.params_yaml)
        prob.params_default = {}
        self.assertIn('no configurable parameters', prob.params_yaml)

    def test_load_failure(self):
        empty_prob = self.tmp_dir / 'problems' / 'PROB2'
        empty_prob.mkdir(parents=True, exist_ok=True)
        module_path = '.'.join(empty_prob.parts[-3:])
        prob = ProblemData('PROB2', [module_path])
        self.assertFalse(prob.available)
        self.assertIn("The subclass of 'MultiTaskProblem' not found.", prob.verbose()['msg'])
        prob = ProblemData('PROB2', ["invalid_path.problems.PROB2"])
        self.assertFalse(prob.available)
        self.assertIn('No module named', '\n'.join(prob.verbose()['msg']))
        with self.assertRaises(ValueError):
            prob.initialize()
        with self.assertRaises(ValueError):
            _ = prob.n_task

    def test_copy(self):
        prob = self.problems.get('PROB1')
        prob_cp = prob.copy()
        self.assertTrue(id(prob) != id(prob_cp))


class TestExperimentConfig(LoadersTestCase):

    def test_default_value(self):
        alg = self.algorithms.get('ALG1')
        prob = self.problems.get('PROB1')
        self.assertTrue(alg.available)
        self.assertTrue(prob.available)
        exp = loaders.ExperimentConfig(alg, prob, 'out/results')
        self.assertIn('Alg', repr(exp))
        self.assertIn('Prob', repr(exp))
        self.assertIsInstance(exp.algorithm, AlgorithmData)
        self.assertIsInstance(exp.problem, ProblemData)
        self.assertIsInstance(exp.root, Path)
        self.assertEqual(exp.root, Path('out/results') / exp.algorithm.name / exp.problem.name / exp.problem.npd_str)
        self.assertIn("FEi", exp.prefix)
        self.assertIn('FEm', exp.prefix)
        self.assertIn('Seed', exp.prefix)
        self.assertIn('Rep01.msgpack', exp.result_name(1).name)

        self.assertEqual(exp.num_results, 0)
        exp.init_root()
        for i in range(5):
            with open(exp.result_name(i), 'w') as f:
                f.write('---')
        self.assertEqual(exp.num_results, 5)

        self.assertIn('algorithm', exp.params_dict)
        self.assertIn('problem', exp.params_dict)
        self.assertIn('PROB', str(exp))
        self.assertIn('ALG', str(exp))

        exp.problem.params_update = {'dim': 2, 'npd': 2}
        prefix = "FEi10_FEm22_Seed123_"
        self.assertEqual(exp.prefix, prefix)
        exp.problem.params_update = {'dim': 2, 'npd': 2, 'fe_init': 6, 'fe_max': 12}
        prefix = "FEi6_FEm12_Seed123_"
        self.assertEqual(exp.prefix, prefix)

        self.assertEqual(exp.num_results, 0)
        exp.init_root()
        for i in range(3):
            with open(exp.result_name(i), 'w') as f:
                f.write('---')
        self.assertEqual(exp.num_results, 3)
        res_root = exp.root.parent.parent
        n_prob = len(os.listdir(res_root))
        self.assertEqual(n_prob, 1, msg=f"res root: {res_root}")

        self.assertEqual(exp.problem.dim, 2)
        self.assertEqual(exp.problem.npd, 2)
        self.assertEqual(exp.problem.dim_str, '2D')
        self.assertEqual(exp.problem.npd_str, 'NPD2')
        self.assertEqual(exp.problem.params['fe_init'], 6)
        self.assertEqual(exp.problem.params['fe_max'], 12)

        exp.create_snapshot([])
        self.assertTrue(exp.root.parent.exists())
        self.assertTrue(exp.code_dest.exists())
        self.assertGreater(len(list(exp.code_dest.iterdir())), 0)
        self.assertTrue(exp.markdown_dest.exists())


class TestLauncherConfig(PyfmtoTestCase):
    def setUp(self):
        self.save_sys_env()
        self.tmp_dir = Path('temp_dir_for_test')
        self.algs = ['alg1', 'alg2']
        self.probs = ['prob1', 'prob2']
        gen_code('algorithms', self.algs, self.tmp_dir)
        gen_code('problems', self.probs, self.tmp_dir)
        filename = gen_config(
            f"""
            launcher:
                sources: [{self.tmp_dir}]
                algorithms: [{self.algs[0]}, {self.algs[1]}]
                problems: [{self.probs[0]}, {self.probs[1]}]
            """,
            self.tmp_dir
        )
        conf = ConfigLoader(filename)
        self.exp = conf.launcher.experiments

    def tearDown(self):
        self.delete('out')
        self.delete(self.tmp_dir)
        self.restore_sys_env()

    def test_valid_config(self):
        config = LauncherConfig(
            results='out/results',
            repeat=3,
            seed=123,
            snapshot=True,
            verbose=False,
            packages=[],
            loglevel='DEBUG',
            save=True,
            algorithms=self.algs,
            problems=self.probs,
            sources=[],
        )
        self.assertEqual(config.results, 'out/results')
        self.assertEqual(config.repeat, 3)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.loglevel, 'DEBUG')
        self.assertEqual(config.packages, [])
        self.assertTrue(config.snapshot)
        self.assertFalse(config.verbose)
        self.assertTrue(config.save)
        self.assertEqual(config.algorithms, self.algs)
        self.assertEqual(config.problems, self.probs)
        self.assertEqual(config.n_exp, 0)
        self.exp[0].success = True
        config.experiments = self.exp
        self.assertEqual(config.n_exp, len(self.probs) * len(self.algs))
        self.assertEqual(config.total_repeat, config.n_exp * config.repeat)
        config.show_summary()


class TestReporterConfig(unittest.TestCase):
    def test_valid_config(self):
        config = ReporterConfig(
            results='out/results',
            algorithms=[['alg1', 'alg2'], ['alg3', 'alg4']],
            problems=['prob1', 'prob2'],
            formats=['curve'],
        )
        self.assertEqual(config.results, 'out/results')
        self.assertEqual(config.algorithms, [['alg1', 'alg2'], ['alg3', 'alg4']])
        self.assertEqual(config.problems, ['prob1', 'prob2'])

    def test_defaults(self):
        config = ReporterConfig(
            algorithms=[['alg1', 'alg2']],
            problems=['prob1'],
            results='out/results',
            formats=['curve'],
        )
        self.assertEqual(config.results, 'out/results')
        self.assertEqual(config.root, Path('out/results'))

    def test_invalid_config(self):
        inner_too_short = {'algorithms': [[]], 'problems': ['prob1']}
        empty_algorithms = {'algorithms': [], 'problems': ['prob1']}
        empty_problems = {'algorithms': [['alg1', 'alg2']], 'problems': []}
        self.assertRaises(ValidationError, ReporterConfig, **inner_too_short)
        self.assertRaises(ValidationError, ReporterConfig, **empty_algorithms)
        self.assertRaises(ValidationError, ReporterConfig, **empty_problems)


class TestConfigLoader(PyfmtoTestCase):

    def setUp(self):
        self.save_sys_env()
        self.tmp_dir = Path('temp_dir_for_test')

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
            algorithms: []
            problems: []
            formats: []
        """

    def tearDown(self):
        self.delete(self.tmp_dir)
        self.restore_sys_env()

    def test_default_config(self):
        filename = gen_config(self.valid, self.tmp_dir)
        conf = ConfigLoader(filename)
        self.assertIn('launcher', conf.config)
        self.assertIn('reporter', conf.config)
        self.assertIn('results', conf.config['reporter'])
        self.assertIn('algorithms', conf.config['reporter'])
        self.assertIn('problems', conf.config['reporter'])
        self.assertIn('formats', conf.config['reporter'])
        l_conf = conf.config['launcher']
        r_conf = conf.config['reporter']
        self.assertEqual(l_conf['results'], r_conf['results'])
        self.assertEqual(l_conf['problems'], r_conf['problems'])
        self.assertEqual(l_conf['algorithms'], ['ALG1', 'ALG2'])
        self.assertEqual(r_conf['algorithms'], [['ALG1', 'ALG2']])
        self.assertEqual(len(conf.config), 2)
        gen_config(
            """
            launcher:
                results: out/launcher/results
                algorithms: [ALG1, ALG2]
                problems: [PROB1, PROB2]
            reporter:
                results: out/reporter/results
            problems:
                PROB1: []
            """,
            self.tmp_dir
        )
        conf = ConfigLoader(filename)
        self.assertEqual(len(conf.config), 3)
        self.assertIn('launcher', conf.config)
        self.assertIn('reporter', conf.config)
        self.assertIn('problems', conf.config)
        self.assertEqual(conf.config['launcher']['results'], 'out/launcher/results')
        self.assertEqual(conf.config['reporter']['results'], 'out/reporter/results')
        self.assertEqual(conf.config['launcher']['algorithms'], ['ALG1', 'ALG2'])
        self.assertEqual(conf.config['reporter']['algorithms'], [['ALG1', 'ALG2']])
        self.assertEqual(conf.config['launcher']['problems'], ['PROB1', 'PROB2'])
        self.assertEqual(conf.config['reporter']['problems'], ['PROB1', 'PROB2'])

    def test_check_config(self):
        gen_code('algorithms', ['ALG1', 'ALG2'], self.tmp_dir)
        gen_code('problems', ['PROB1', 'PROB2'], self.tmp_dir)
        filename = gen_config(self.invalid, self.tmp_dir)
        conf = ConfigLoader(filename)
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')

        gen_config(
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
                algorithms: [123, ALG1, ALG2, [ALG3, ALG4]]
                problems: []
                formats: []
            """,
            self.tmp_dir
        )
        conf = ConfigLoader(filename)
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')

    def test_conf_values(self):
        filename = gen_config(self.valid, self.tmp_dir)
        conf = ConfigLoader(filename)
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 0)
        self.assertEqual(conf.reporter.experiments, [])

        gen_code('algorithms', ['ALG1', 'ALG2'], self.tmp_dir)
        gen_code('problems', ['PROB1', 'PROB2'], self.tmp_dir)
        conf = ConfigLoader(filename)
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 4)
        self.assertEqual(len(conf.reporter.experiments), 4)

    def test_show_sources(self):
        gen_code('algorithms', ['ALG1', 'ALG2'], self.tmp_dir)
        gen_code('problems', ['PROB1', 'PROB2'], self.tmp_dir)
        conf_file = gen_config(
            f"""
            launcher:
                sources: [{self.tmp_dir}]
            """,
            self.tmp_dir
        )
        conf = ConfigLoader(conf_file)
        alg_str = conf.show_sources('algorithms', True)
        prob_str = conf.show_sources('problems', True)
        self.assertIn('ALG1', alg_str)
        self.assertIn('ALG2', alg_str)
        self.assertIn('PROB1', prob_str)
        self.assertIn('PROB2', prob_str)
        self.assertNotIn('ALG1', prob_str)
        self.assertNotIn('ALG2', prob_str)
        self.assertNotIn('PROB1', alg_str)
        self.assertNotIn('PROB2', alg_str)
