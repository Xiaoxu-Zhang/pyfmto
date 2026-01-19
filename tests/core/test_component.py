import unittest
from pyfmto.core import ComponentData


class TestComponent(unittest.TestCase):
    def test_default(self):
        component = ComponentData()
        self.assertIsInstance(component.desc, dict)
        self.assertNotEqual(str(component), '')
        self.assertNotEqual(repr(component), '')
        self.assertFalse(component.available)
        self.assertIn('not available', component.params_yaml)
        self.assertEqual(component._parse_default_params('no_exist'), {})
