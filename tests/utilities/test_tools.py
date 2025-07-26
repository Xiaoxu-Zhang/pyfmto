import shutil
import unittest
import time
from pathlib import Path

from pyfmto.utilities import tabulate_formats
from pyfmto.utilities.tools import colored, timer, show_in_table, titled_tabulate, update_kwargs


class TestTools(unittest.TestCase):

    def setUp(self):
        self.out_path = Path('out')

    def tearDown(self):
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_colored(self):
        text = "test"
        color_text = colored(text, 'green')
        self.assertIn('\033[32m', color_text)
        self.assertIn(text, color_text)
        self.assertRaises(ValueError, colored, text, 'invalid_color')

    def test_timer(self):
        @timer()
        def dummy_func1():
            time.sleep(0.1)

        @timer(name='dummy_func', where='console')
        def dummy_func2():
            time.sleep(0.1)

        @timer(where='log')
        def dummy_func3():
            time.sleep(0.1)

        @timer(where='both')
        def dummy_func4():
            time.sleep(0.1)

        dummy_func1()
        dummy_func2()
        dummy_func3()
        dummy_func4()

    def test_show_in_table(self):
        settings = {
            "verbose": True,
            "iterations": 100,
            "learning_rate": 0.01,
            "use_cache": False,
            "description": "Test Settings"
        }
        colored_table, _ = show_in_table(**settings)
        self.assertIn("yes", colored_table)
        self.assertIn("no", colored_table)
        self.assertIn("\033[32m", colored_table)  # green for float
        self.assertIn("\033[31m", colored_table)  # red for boolean (False)
        self.assertIn("\033[35m", colored_table)  # magenta for int

    def test_titled_tabulate(self):
        data = {"abc": [1, 2, 3], "bcd": [4, 5, 6], 'cde': [7, 8, 9]}
        tit = titled_tabulate(
            "Test",
            '=',
            data,
            headers="keys",
            tablefmt=tabulate_formats.rounded_grid
        )

        lines = tit.split('\n')
        self.assertTrue('Test' in lines[1], f"Titled table is \n {tit}")
        self.assertEqual(len(tit[1]), len(tit[2]))

    def test_warn_unused_kwargs(self):
        empty1 = update_kwargs('test', {}, {'a': 1, 'b': 2})
        empty2 = update_kwargs('test', {}, {})
        res1 = update_kwargs('test', {'a': 1, 'b': 2}, {})
        res2 = update_kwargs('test', {'a': 1, 'b': 2}, {'a': 2})

        self.assertEqual(empty1, {})
        self.assertEqual(empty2, {})
        self.assertTrue(res1 == {'a': 1, 'b': 2})
        self.assertTrue(res2 == {'a': 2, 'b': 2})
