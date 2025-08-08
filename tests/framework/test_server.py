import pickle
import unittest
from unittest.mock import AsyncMock

from pyfmto.framework.server import load_body
from tests.framework import OfflineServer


class TestEmptyServer(unittest.TestCase):

    def test_attributes(self):
        server = OfflineServer()
        self.assertEqual(server.num_clients, 0)
        self.assertTrue(list(server.sorted_ids) == [])
        self.assertEqual(server.alpha, 0.1)
        self.assertEqual(server.beta, 0.2)

    def test_configurable_server(self):
        class ConfigurableServer(OfflineServer):
            """
            alpha: 0.1
            beta: 0.2
            """
            def __init__(self, **kwargs):
                super().__init__()
                kwargs = self.update_kwargs(kwargs)
                self.alpha = kwargs['alpha']
                self.beta = kwargs['beta']
        server = ConfigurableServer(alpha=0.3, beta=0.4)
        self.assertEqual(server.alpha, 0.3)
        self.assertEqual(server.beta, 0.4)

    def test_methods(self):
        server = OfflineServer()

        self.assertEqual(server._config.host, 'localhost')
        self.assertEqual(server._config.port, 18510)
        self.assertEqual(server._agg_interval, 0.5)

        for ip, port in zip(['1.2.3.4', '5.6.7.8', '9.10.11.12'], [18510, 18511, 18512]):
            server.set_addr(ip, port)
            self.assertEqual(server._config.host, ip)
            self.assertEqual(server._config.port, port)

        for agg_itv in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            server.set_agg_interval(agg_itv)
            self.assertEqual(server._agg_interval, max(agg_itv, 0.01))

        server.enable_consol_log()


class TestLoadBody(unittest.IsolatedAsyncioTestCase):
    async def test_load_body_empty_body(self):
        mock_request = AsyncMock()
        mock_request.body.return_value = b""
        result = await load_body(mock_request)
        self.assertIsNone(result)

    async def test_load_body_valid_pickle_data(self):
        mock_request = AsyncMock()
        test_data = {"key": "value"}
        mock_request.body.return_value = pickle.dumps(test_data)
        result = await load_body(mock_request)
        self.assertEqual(result, test_data)

    async def test_load_body_invalid_pickle_data(self):
        mock_request = AsyncMock()
        mock_request.body.return_value = b"invalid_pickle_data"
        with self.assertRaises(pickle.UnpicklingError):
            await load_body(mock_request)
