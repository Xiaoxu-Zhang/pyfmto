from tests.helpers.testcases import TestCaseAlgProbConf


class TestAlgorithmData(TestCaseAlgProbConf):

    def test_default_value(self):
        alg_name = self.alg_names[0]
        alg = self.load_algorithm(alg_name)
        self.assertEqual(alg.params_default, {'client': {'name': 'c'}, 'server': {'name': 's'}})
        self.assertEqual(alg.params_default, alg.params)
        self.assertEqual(alg.name, alg_name)
        self.assertEqual(alg.name_alias, '')
        self.assertTrue('client' in alg.params_yaml)
        self.assertNotEqual(alg.params_snapshot, '')

    def test_update_params(self):
        alg_name = self.alg_names[0]
        alg = self.load_algorithm(alg_name)
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
