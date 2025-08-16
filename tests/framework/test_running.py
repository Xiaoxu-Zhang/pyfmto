import threading
import time
import unittest

from pyfmto.experiments.utils import LauncherUtils
from pyfmto.framework import Client, Server, ClientPackage, ServerPackage
from pyfmto.problems import load_problem
from tests.framework import (
    OfflineServer, OnlineClient, OnlineServer
)
from requests.exceptions import ConnectionError

N_CLIENTS = 2


class TestClientSide(unittest.TestCase):

    def setUp(self):
        self.problems = load_problem('tetci2019', dim=5, fe_init=20, fe_max=30)
        self.utils = LauncherUtils

    def test_valid_offline_client(self):
        """An offline client doesn't request the server"""
        class OfflineClient(Client):
            def optimize(self):
                time.sleep(0.01)
                x = self.problem.random_uniform_x(1)
                y = self.problem.evaluate(x)
                self.problem.solutions.append(x, y)
        clients = [OfflineClient(prob) for prob in self.problems[:N_CLIENTS]]
        with self.utils.running_server(OfflineServer):
            self.utils.start_clients(clients)

    def test_valid_online_client(self):
        clients = [OnlineClient(prob) for prob in self.problems[:N_CLIENTS]]
        with self.utils.running_server(OfflineServer):
            with self.assertRaises(ConnectionError):
                self.utils.start_clients(clients)

    def test_request_failed(self):
        client = OnlineClient(self.problems[0])
        with self.assertRaises(ConnectionError):
            self.utils.start_clients([client])

    def test_invalid_clients(self):

        class ClientWithOptimizeErr(Client):
            def optimize(self):
                raise RuntimeError("Test raise error")

        class ClientWithRequestErr(Client):
            def optimize(self):
                self.request_server(None)

        clients = [ClientWithOptimizeErr(prob) for prob in self.problems[:N_CLIENTS]]
        with self.utils.running_server(OnlineServer):
            with self.assertRaises(RuntimeError):
                self.utils.start_clients(clients)

        clients = [ClientWithRequestErr(prob) for prob in self.problems[:N_CLIENTS]]
        with self.utils.running_server(OnlineServer):
            with self.assertRaises(ValueError):
                self.utils.start_clients(clients)

    def test_valid_server(self):
        server = OnlineServer()
        problems = load_problem('tetci2019', dim=5, fe_init=20, fe_max=30)
        clients = [OnlineClient(prob) for prob in problems[:N_CLIENTS]]
        thread = threading.Thread(target=server.start)
        thread.start()
        time.sleep(1)
        self.utils.start_clients(clients)
        server.shutdown()

    def test_invalid_server_agg(self):
        class InvalidServerHandler(Server):
            def handle_request(self, client_data: ClientPackage) -> ServerPackage:
                raise RuntimeError('Test exception in server.handle_request()')

            def aggregate(self):
                raise RuntimeError("Test exception in server.aggregate()")

        class InvalidServerAgg(Server):
            def handle_request(self, client_data: ClientPackage) -> ServerPackage:
                pass

            def aggregate(self):
                raise RuntimeError("Test raise error")

        server = InvalidServerHandler()
        thread = threading.Thread(target=server.start)
        thread.start()
        thread.join(timeout=1)

        server = InvalidServerAgg()
        thread = threading.Thread(target=server.start)
        thread.start()
        thread.join(timeout=1)
