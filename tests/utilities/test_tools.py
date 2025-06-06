import unittest
import time

from pyfmto.utilities.tools import colored, timer, show_in_table


class TestTools(unittest.TestCase):

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
