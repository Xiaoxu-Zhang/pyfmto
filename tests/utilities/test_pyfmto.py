import unittest
from pathlib import Path

from pyfmto import export_default_config, export_launcher_config, export_reporter_config, export_algorithm_config, \
    export_problem_config


class TestPyfmto(unittest.TestCase):
    def setUp(self):
        self.conf = Path('config.yaml')
        self.tmp_files = []

    def tearDown(self):
        if self.conf.exists():
            self.conf.unlink()
        for p in self.tmp_files:
            if p.exists():
                p.unlink()

    def test_export_default_config(self):
        export_default_config()
        self.assertTrue(self.conf.exists())

    def test_export_to_new(self):
        funcs = [
            export_launcher_config,
            export_reporter_config,
            export_algorithm_config,
            export_problem_config,
        ]

        # Repeat twice to cover file exists case
        self.tmp_files = [f() for f in funcs+funcs]
        for p in self.tmp_files:
            self.assertTrue(p.exists())

    def test_export_invalid_config(self):
        export_algorithm_config(algs=['INVALID'])
        export_problem_config(probs=['INVALID'])