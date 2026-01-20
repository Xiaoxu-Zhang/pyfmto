from itertools import product

import numpy as np

from pyfmto.framework import Client
from pyfmto.utilities.io import parse_yaml
from tests.helpers.testcases import TestCaseAlgProbConf


class TestClient(TestCaseAlgProbConf):
    def setUp(self):
        super().setUp()
        self.problem = self.load_problem(self.prob_names[0]).initialize()

    def test_empty_client_attributes(self):

        class EmptyClient(Client):
            def optimize(self):
                pass

        task = self.problem[0]
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
                self.alpha = kwargs['alpha']
                self.beta = kwargs['beta']

            def optimize(self):
                pass

        task = self.problem[0]
        defaults = parse_yaml(ConfigurableClient.__doc__)
        default_config = ConfigurableClient(task, **defaults)
        for k, v in defaults.items():
            self.assertEqual(getattr(default_config, k), v, f'Client parameter {k} is not equal to {v}')
        for alpha, beta in product([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]):
            with self.subTest(alpha=alpha, beta=beta):
                client = ConfigurableClient(task, alpha=alpha, beta=beta)
                self.assertEqual(client.alpha, alpha, f'Client parameter alpha is not equal to {alpha}')
                self.assertEqual(client.beta, beta, f'Client parameter beta is not equal to {beta}')
