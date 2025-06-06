import os
import unittest
import yaml
from unittest import mock


settings = """
algorithms: [ALG1, ALG2]
problems:
    prob1:
        class: Prob1Class
        args:
            single: 10
            multi_int: [1, 2]
    prob2:
        class: Prob2Class
        args:
            single_int: 10
            multi_int: [1, 2, 4, 6]
            multi_str: [Ackley, Griewank]
"""

class TestRunner(unittest.TestCase):

    def setUp(self):
        self.os_name  = os.name
        self.settings = yaml.safe_load(settings)

    @mock.patch('os.name', 'posix')
    def test_linux_start_server(self):
        self.testing_server()

    @mock.patch('os.name', 'nt')
    def test_win_start_server(self):
        self.testing_server()

    def testing_server(self):
        if self.is_mocking:
            msg = f"test by mocking os.name to {self.os_name}"
            # test run and catch exception
            pass
        else:
            msg = f"test by running on real os {self.os_name}"
            # test run and stop # on a real system
            pass

    @property
    def is_mocking(self):
        return os.name != self.os_name