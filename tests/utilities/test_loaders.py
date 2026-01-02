import copy
import os
import unittest
from pathlib import Path
from pydantic import ValidationError
from ruamel.yaml import CommentedMap

from pyfmto.problems import MultiTaskProblem
from pyfmto.utilities import loaders, save_yaml
from pyfmto.experiments import list_report_formats, show_default_conf
from pyfmto.framework import Client, Server
from pyfmto.utilities.loaders import (
    ProblemData, AlgorithmData,
    LauncherConfig, ReporterConfig,
    DataLoader
)
from tests.helpers import remove_temp_files, gen_algorithm, gen_problem


class TestOtherHelpers(unittest.TestCase):

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


class TestAlgorithmHelpers(unittest.TestCase):

    def tearDown(self):
        remove_temp_files()

    def test_list_without_alg_dir(self):
        res = loaders.list_algorithms(print_it=True)
        self.assertEqual(res, {})

    def test_list_algorithms(self):
        algs = ['ALG1', 'ALG2', 'ALG3']
        gen_algorithm(algs)

        # Make an invalid algorithm to cover the load failure lines
        Path('algorithms', 'INVALID').mkdir(parents=True, exist_ok=True)

        res = loaders.list_algorithms(print_it=True)
        self.assertEqual(set(res.keys()), set(algs))

    def test_load_algorithm(self):
        gen_algorithm('ALG1')
        res = loaders.load_algorithm('ALG1')
        self.assertTrue(issubclass(res.client, Client))
        self.assertTrue(issubclass(res.server, Server))
        self.assertEqual(res.params_default, {'client': {'name': 'c'}, 'server': {'name': 's'}})

    def test_load_no_algorithms_dir(self):
        with self.assertRaises(FileNotFoundError):
            loaders.load_algorithm('ALG')

    def test_load_invalid_algorithm(self):
        Path('algorithms', 'INVALID').mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ModuleNotFoundError):
            loaders.load_algorithm('INVALID')
        with self.assertRaises(ValueError):
            loaders.load_algorithm('NONEXISTENT')

    def test_list_report_formats(self):
        res = list_report_formats(print_it=True)
        for fmt in ['curve', 'violin', 'excel', 'latex', 'console']:
            self.assertIn(fmt, res)

    def test_show_default_conf(self):
        for fmt in list_report_formats() + ['nonexist']:
            with self.subTest(fmt=fmt):
                self.assertIsNone(show_default_conf(fmt))


class TestProblemHelpers(unittest.TestCase):

    def setUp(self):
        self.n_probs = len(loaders.list_problems(print_it=True))
        self.probs = ['PROB1', 'PROB2']
        gen_problem(self.probs)

    def tearDown(self):
        remove_temp_files()

    def test_list_problems(self):
        res2 = loaders.list_problems(print_it=True)
        self.assertEqual(self.n_probs, len(res2) - len(self.probs))
        for prob in self.probs:
            with self.subTest(prob=prob):
                self.assertIn(prob, res2)

    def test_load_problem(self):
        res = loaders.load_problem(self.probs[0])
        self.assertIsInstance(res, ProblemData)
        with self.assertRaises(ValueError):
            loaders.load_problem('NONEXISTENT')

    def test_init_problem(self):
        res = loaders.init_problem(self.probs[0])
        self.assertIsInstance(res, MultiTaskProblem)


class TestAlgorithmData(unittest.TestCase):

    def setUp(self):
        gen_algorithm('ALG1')

    def tearDown(self):
        remove_temp_files()

    def test_default_value(self):
        alg = loaders.load_algorithm('ALG1')
        self.assertEqual(alg.params_default, {'client': {'name': 'c'}, 'server': {'name': 's'}})
        self.assertEqual(alg.params_default, alg.params)
        self.assertEqual(alg.name, 'ALG1')
        self.assertEqual(alg.name_alias, '')
        self.assertTrue('client' in alg.params_yaml)

    def test_update_params(self):
        alg = loaders.load_algorithm('ALG1')
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

    def test_copy(self):
        alg = loaders.load_algorithm('ALG1')
        alg_cp = alg.copy()
        self.assertTrue(id(alg) != id(alg_cp))
        self.assertEqual(alg.__dict__, alg_cp.__dict__)


class TestProblemData(unittest.TestCase):
    def setUp(self):
        gen_problem('PROB1')

    def tearDown(self):
        remove_temp_files()

    def test_default_value(self):
        prob = loaders.load_problem('PROB1')
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

    def test_copy(self):
        prob = loaders.load_problem('PROB1')
        prob_cp = prob.copy()
        self.assertTrue(id(prob) != id(prob_cp))


class TestExperimentConfig(unittest.TestCase):

    def setUp(self):
        gen_algorithm('ALG1')
        gen_problem('PROB1')
        self.prob = loaders.load_problem('PROB1')
        self.alg = loaders.load_algorithm('ALG1')
        self.prob.params_default.pop('dim')

    def tearDown(self):
        remove_temp_files()

    def test_default_value(self):
        exp = loaders.ExperimentConfig(self.alg, self.prob, 'out/results')
        self.assertIn('Alg', repr(exp))
        self.assertIn('Prob', repr(exp))
        self.assertIsInstance(exp.algorithm, AlgorithmData)
        self.assertIsInstance(exp.problem, ProblemData)
        self.assertIsInstance(exp.root, Path)
        self.assertEqual(exp.root, Path('out/results') / exp.algorithm.name / exp.problem.name / exp.problem.npd_str)
        prefix = f"Seed{exp.problem.params['seed']}_"
        self.assertEqual(exp.prefix, prefix)
        self.assertEqual(exp.result_name(1).name, f'{prefix}Rep01.msgpack')

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
        self.assertEqual(n_prob, 2, msg=f"res root: {res_root}")

        self.assertEqual(exp.problem.dim, 2)
        self.assertEqual(exp.problem.npd, 2)
        self.assertEqual(exp.problem.dim_str, '2D')
        self.assertEqual(exp.problem.npd_str, 'NPD2')
        self.assertEqual(exp.problem.params['fe_init'], 6)
        self.assertEqual(exp.problem.params['fe_max'], 12)

        exp.backup_params()
        self.assertTrue(exp.root.parent.exists())
        self.assertTrue((exp.root.parent / 'parameters.yaml').exists())


class TestLauncherConfig(unittest.TestCase):
    def setUp(self):
        gen_algorithm('ALG1')
        for prob in ['PROB1', 'PROB2', 'PROB3']:
            gen_problem(prob)
        self.exp = [
            loaders.ExperimentConfig(loaders.load_algorithm('ALG1'), loaders.load_problem(prob), 'out/results')
            for prob in ['PROB1', 'PROB2', 'PROB3']
        ]

    def tearDown(self):
        remove_temp_files()

    def test_valid_config(self):
        config = LauncherConfig(
            results='out/results',
            repeat=3,
            seed=123,
            backup=True,
            loglevel='DEBUG',
            save=True,
            algorithms=['alg1', 'alg2'],
            problems=['prob1', 'prob2']
        )
        self.assertEqual(config.results, 'out/results')
        self.assertEqual(config.repeat, 3)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.loglevel, 'DEBUG')
        self.assertTrue(config.backup)
        self.assertTrue(config.save)
        self.assertEqual(config.algorithms, ['alg1', 'alg2'])
        self.assertEqual(config.problems, ['prob1', 'prob2'])
        self.assertEqual(config.n_exp, 0)
        self.exp[0].success = True
        config.experiments = self.exp
        self.assertEqual(config.n_exp, 3)
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


class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('out')
        self.conf_dir = self.tmp_dir / 'config.yaml'
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.valid = {
            'launcher': {
                'algorithms': ['ALG1', 'ALG2'],
                'problems': ['PROB1', 'PROB2']
            },
        }
        self.invalid = {
            'launcher': {
                'results': '',
                'repeat': 0,
                'save': 'yes',
                'loglevel': 'INVALID',
                'algorithms': [],
                'problems': [],
            },
            'reporter': {
                'results': '',
                'algorithms': [],
                'problems': [],
                'formats': [],
            }
        }

    def tearDown(self):
        remove_temp_files()

    def make_conf(self, conf: dict):
        save_yaml(conf, self.conf_dir)

    def load_conf(self) -> DataLoader:
        return DataLoader(str(self.conf_dir))

    def copy_valid_conf(self):
        return copy.deepcopy(self.valid)

    def copy_invalid_conf(self):
        return copy.deepcopy(self.invalid)

    def test_default_config(self):
        self.make_conf({})
        conf = self.load_conf()
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
        self.assertEqual(l_conf['algorithms'], [])
        self.assertEqual(r_conf['algorithms'], [[]])
        self.assertEqual(len(conf.config), 2)

        self.make_conf(
            {
                'launcher': {
                    'results': 'out/launcher/results',
                    'algorithms': ['ALG1', 'ALG2'],
                    'problems': ['PROB1', 'PROB2']
                },
                'reporter': {
                    'results': 'out/reporter/results',
                },
                'problems': {}
            }
        )
        conf = self.load_conf()
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

        self.make_conf(self.invalid)
        conf = self.load_conf()
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')

        conf_src = self.copy_invalid_conf()
        conf_src['reporter']['algorithms'] = [123, 'ALG1', 'ALG2', ['ALG3', 'ALG4']]
        self.make_conf(conf_src)
        conf = self.load_conf()
        with self.assertRaises(ValueError):
            conf.check_config_issues('reporter')
        with self.assertRaises(ValueError):
            conf.check_config_issues('launcher')

    def test_conf_values(self):
        self.make_conf(self.valid)
        conf = self.load_conf()
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 0)
        self.assertEqual(conf.reporter.experiments, [])

        gen_algorithm(conf.config['launcher']['algorithms'])
        gen_problem(conf.config['launcher']['problems'])
        self.assertIsInstance(conf.launcher, LauncherConfig)
        self.assertIsInstance(conf.reporter, ReporterConfig)
        self.assertEqual(conf.launcher.n_exp, 4)
        self.assertEqual(len(conf.reporter.experiments), 4)
