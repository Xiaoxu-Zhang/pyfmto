import unittest

from pyfmto.experiments import kill_server
from pyfmto.experiments.utils import start_server, start_clients
from pyfmto.problems import load_problem
from tests.framework import OfflineClient, OfflineServer, OnlineClient, OnlineServer, InvalidClient
from requests.exceptions import ConnectionError

N_CLIENTS = 3


class TestValidRuns(unittest.TestCase):

    def setUp(self):
        self.problems = load_problem('tetci2019', dim=5, fe_init=20, fe_max=25)

    def test_valid_offline_client(self):
        clients = [OfflineClient(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OfflineServer)
        start_clients(clients)

    def test_valid_online_client(self):
        clients = [OnlineClient(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OfflineServer)
        start_clients(clients)

        clients = [OnlineClient(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OnlineServer)
        start_clients(clients)

    def test_request_failed(self):
        client = OnlineClient(self.problems[0])
        kill_server()
        with self.assertRaises(ConnectionError):
            start_clients([client])

    def test_invalid_client(self):
        clients = [InvalidClient(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OnlineServer)
        with self.assertRaises(RuntimeError):
            start_clients(clients)
