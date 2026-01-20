import numpy as np
from ruamel.yaml.error import MarkedYAMLError

from pyfmto.utilities.io import dumps_yaml, load_msgpack, load_yaml, parse_yaml, save_msgpack, save_yaml, \
    recursive_to_pure_dict, _to_builtin_type
from tests.helpers import PyfmtoTestCase

YAML_OK = """
key1:
  key11: value1

  key12: value2

key2:
  key21: value3
  key22: value4
"""

YAML_BAD = """
key: [value
"""


class TestYaml(PyfmtoTestCase):

    def setUp(self):
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_ok = self.tmp_dir / 'yaml_ok.yaml'
        self.yaml_bad = self.tmp_dir / 'yaml_bad.yaml'
        self.not_exists = self.tmp_dir / 'not_exists.yaml'
        self.create_yaml_files()

    def create_yaml_files(self):
        with open(self.yaml_ok, 'w') as f:
            f.write(YAML_OK)
        with open(self.yaml_bad, 'w') as f:
            f.write(YAML_BAD)

    def test_save_yaml(self):
        save_yaml({'name': 'save_yaml'}, self.tmp_dir / 'save_yaml.yaml')
        self.assertTrue((self.tmp_dir / 'save_yaml.yaml').exists())

    def test_parse_yaml(self):
        self.assertEqual(parse_yaml(None), {})
        self.assertEqual(parse_yaml(''), {})
        self.assertTrue('key1' in parse_yaml(YAML_OK))

    def test_dumps_yaml(self):
        res = dumps_yaml(parse_yaml(YAML_OK))
        self.assertEqual(len(res.splitlines()), 7, f"res is {res}")

    def test_load_yaml(self):
        data = load_yaml(self.yaml_ok)
        self.assertIsNotNone(data)
        self.assertRaises(MarkedYAMLError, load_yaml, self.yaml_bad)
        self.assertRaises(FileNotFoundError, load_yaml, self.not_exists)
        self.assertEqual(load_yaml(self.not_exists, True), {})


class TestMsgpack(PyfmtoTestCase):

    def test_pack_with_ndarray(self):
        data = {
            'name': 'pack_with_ndarray',
            'data': np.array([1, 2, 3])
        }
        pack_with_ndarray = self.tmp_dir / 'pack_with_ndarray.msgpack'
        save_msgpack(data, pack_with_ndarray)
        self.assertTrue(pack_with_ndarray.exists())
        load_data = load_msgpack(pack_with_ndarray)
        self.assertTrue(np.array_equal(load_data['data'], data['data']))
        self.assertTrue(load_data['name'] == data['name'])

    def test_pack_with_set(self):
        data = {
            'name': 'pack_with_set',
            'data': {1, 2, 3}
        }
        pack_with_set = self.tmp_dir / 'pack_with_set.msgpack'
        save_msgpack(data, pack_with_set)
        self.assertTrue(pack_with_set.exists())
        load_data = load_msgpack(pack_with_set)
        self.assertTrue(load_data['data'] == data['data'])
        self.assertTrue(load_data['name'] == data['name'])

    def test_recursive_pack(self):
        data = {
            'name': 'pack_recursive',
            'data': {
                'evaluation': {
                    'x': np.array([1, 2, 3]),
                    'y': 0.5
                },
                'ids': {1, 2, 3}
            }
        }
        pack_recursive_dict = self.tmp_dir / 'pack_recursive.msgpack'
        save_msgpack(data, pack_recursive_dict)
        self.assertTrue(pack_recursive_dict.exists())
        load_data = load_msgpack(pack_recursive_dict)
        self.assertTrue(np.array_equal(load_data['data']['evaluation']['x'], data['data']['evaluation']['x']))
        self.assertTrue(load_data['data']['evaluation']['y'] == data['data']['evaluation']['y'])
        self.assertTrue(load_data['data']['ids'] == data['data']['ids'])

    def test_unsupported_type(self):
        data = {
            'func': lambda x: x
        }
        pack_unsupported_type = self.tmp_dir / 'pack_unsupported_type.msgpack'
        self.assertRaises(TypeError, save_msgpack, data, pack_unsupported_type)


class TestRecursiveToPureDict(PyfmtoTestCase):
    def test_recursive_pure_dict(self):
        yml = """
        a: x  # comment1
        b: [1, 2]  # comment2
        c: [a, b]  # comment3
        d: false  # comment4
        e:  # comment5
          d1: 4  # comment6
          d2:  # comment7
          - abc  # comment8
          - 1  # comment9
        """
        yml_dict = parse_yaml(yml)
        pure_dict = recursive_to_pure_dict(yml_dict)
        self.assertEqual(type(pure_dict['a']), type(''))
        self.assertEqual(type(pure_dict['b']), type([]))
        self.assertEqual(type(pure_dict['b'][0]), type(0))
        self.assertEqual(type(pure_dict['c']), type([]))
        self.assertEqual(type(pure_dict['c'][0]), type(''))
        self.assertEqual(type(pure_dict['d']), type(True))
        self.assertEqual(type(pure_dict['e']), type({}))
        self.assertEqual(type(pure_dict['e']['d1']), type(0))
        self.assertEqual(type(pure_dict['e']['d2']), type([]))
        self.assertEqual(type(pure_dict['e']['d2'][0]), type(''))
        self.assertEqual(type(pure_dict['e']['d2'][1]), type(0))

    def test_to_builtin_type(self):
        class NoBuiltinType:
            def __init__(self):
                pass
        self.assertEqual(type(_to_builtin_type(NoBuiltinType())), type(NoBuiltinType()))
