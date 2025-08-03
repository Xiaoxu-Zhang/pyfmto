import unittest
from concurrent.futures import ThreadPoolExecutor

from pyfmto.experiments import kill_server
from pyfmto.experiments.utils import start_server, start_clients
from pyfmto.framework import Client
from pyfmto.problems import load_problem
from tests.framework import (
    start_subprocess_clients, OfflineClient, OfflineServer, OnlineClient, OnlineServer,
    InvalidServerAgg
)
from requests.exceptions import ConnectionError

N_CLIENTS = 2


class TestClientSide(unittest.TestCase):

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
        with self.assertRaises(ConnectionError):
            start_clients(clients)

    def test_request_failed(self):
        client = OnlineClient(self.problems[0])
        kill_server()
        with self.assertRaises(ConnectionError):
            start_clients([client])

    def test_invalid_clients(self):

        class ClientWithOptimizeErr(Client):
            def optimize(self):
                raise RuntimeError("Test raise error")

        class ClientWithRequestErr(Client):
            def optimize(self):
                self.request_server(None)

        clients = [ClientWithOptimizeErr(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OnlineServer)
        with self.assertRaises(RuntimeError):
            start_clients(clients)

        clients = [ClientWithRequestErr(prob) for prob in self.problems[:N_CLIENTS]]
        kill_server()
        start_server(OnlineServer)
        with self.assertRaises(ValueError):
            start_clients(clients)



class TestServerSide(unittest.TestCase):
    def setUp(self):
        kill_server()

    def test_valid_server(self):
        server = OnlineServer()
        thread_pool = ThreadPoolExecutor(max_workers=1)
        thread_pool.submit(server.start)
        start_subprocess_clients(OnlineClient)
        thread_pool.shutdown(wait=True)


class TestInvalidServerAgg(unittest.TestCase):
    def test_invalid_server_agg(self):
        server = InvalidServerAgg()
        thread_pool = ThreadPoolExecutor(max_workers=1)
        thread_pool.submit(server.start)
        thread_pool.shutdown(wait=True)