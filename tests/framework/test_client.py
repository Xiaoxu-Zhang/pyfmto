import numpy as np
import unittest
from itertools import product

from pyfmto.framework import Client
from pyfmto import init_problem
from pyfmto.utilities import parse_yaml
from tests.helpers import gen_problem


class TestClient(unittest.TestCase):
    def setUp(self):
        gen_problem('PROB')
        self.problems = init_problem('PROB')

    def tearDown(self):
        from tests.helpers import remove_temp_files
        remove_temp_files()

    def test_empty_client_attributes(self):

        class EmptyClient(Client):
            def optimize(self):
                pass

        task = self.problems[0]
        client = EmptyClient(task)
        self.assertEqual(client.name, f"Client {task.id: <2d}")
        self.assertEqual(client.id, task.id)
        self.assertEqual(client.dim, task.dim)
        self.assertEqual(client.obj, task.obj)
        self.assertEqual(client.fe_init, task.fe_init)
        self.assertEqual(client.fe_max, task.fe_max)
        self.assertEqual(client.y_min, task.solutions.y_min)
        self.assertEqual(client.y_max, task.solutions.y_max)
        self.assertTrue(np.all(client.lb == task.lb))
        self.assertTrue(np.all(client.ub == task.ub))

    def test_configurable_client(self):
        class ConfigurableClient(Client):
            """
            alpha: 0.1
            beta: 0.2
            """

            def __init__(self, problem, **kwargs):
                super().__init__(problem)
                kwargs = self.update_kwargs(kwargs)
                self.alpha = kwargs['alpha']
                self.beta = kwargs['beta']

            def optimize(self):
                pass

        prob = self.problems[0]
        defaults = parse_yaml(ConfigurableClient.__doc__)
        default_config = ConfigurableClient(prob)
        for k, v in defaults.items():
            self.assertEqual(getattr(default_config, k), v, f'Client parameter {k} is not equal to {v}')
        for alpha, beta in product([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]):
            with self.subTest(alpha=alpha, beta=beta):
                client = ConfigurableClient(prob, alpha=alpha, beta=beta)
                self.assertEqual(client.alpha, alpha, f'Client parameter alpha is not equal to {alpha}')
                self.assertEqual(client.beta, beta, f'Client parameter beta is not equal to {beta}')
