import numpy as np
import shutil
import unittest
from pathlib import Path
from ruamel.yaml.error import MarkedYAMLError

from pyfmto.utilities import save_yaml, dumps_yaml, parse_yaml
from pyfmto.utilities.io import load_yaml, save_msgpack, load_msgpack

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


class TestYaml(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir()
        self.yaml_ok = self.tmp_dir / 'yaml_ok.yaml'
        self.yaml_bad = self.tmp_dir / 'yaml_bad.yaml'
        self.not_exists = self.tmp_dir / 'not_exists.yaml'
        self.create_yaml_files()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

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


class TestMsgpack(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path('tmp')
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir()

    def tearDown(self):
        shutil.rmtree('tmp')

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
