import threading
import time

from requests.exceptions import ConnectionError

from pyfmto import load_problem
from pyfmto.framework import Client, ClientPackage, Server
from tests.framework import OfflineServer, OnlineClient, OnlineServer
from tests.helpers import running_server, start_clients
from tests.helpers.testcases import TestCaseAlgProbConf

N_CLIENTS = 2


class TestClientSide(TestCaseAlgProbConf):

    def setUp(self):
        super().setUp()
        self.problem = self.load_problem(self.prob_names[0], dim=5, fe_init=20, fe_max=30).initialize()

    def test_valid_offline_client(self):
        """An offline client doesn't request the server"""
        class OfflineClient(Client):
            def optimize(self):
                time.sleep(0.01)
                x = self.problem.random_uniform_x(1)
                y = self.problem.evaluate(x)
                self.problem.solutions.append(x, y)
        clients = [OfflineClient(prob) for prob in self.problem[:N_CLIENTS]]
        with running_server(OfflineServer):
            start_clients(clients)

        clients = [OnlineClient(prob) for prob in self.problem[:N_CLIENTS]]
        with running_server(OnlineServer):
            start_clients(clients)

    def test_request_failed(self):
        client = OnlineClient(self.problem[0])
        with self.assertRaises(ConnectionError):
            start_clients([client])

    def test_invalid_clients(self):

        class ClientWithOptimizeErr(Client):
            def optimize(self):
                raise RuntimeError("Test raise error")

        class ClientWithRequestErr(Client):
            def optimize(self):
                self.request_server(None)

        clients = [ClientWithOptimizeErr(prob) for prob in self.problem[:N_CLIENTS]]
        with running_server(OnlineServer):
            with self.assertRaises(RuntimeError):
                start_clients(clients)

        clients = [ClientWithRequestErr(prob) for prob in self.problem[:N_CLIENTS]]
        with running_server(OnlineServer):
            with self.assertRaises(ValueError):
                start_clients(clients)

    def test_valid_server(self):
        server = OnlineServer()
        problem = load_problem(self.prob_names[0], self.sources, dim=5, fe_init=20, fe_max=30).initialize()
        clients = [OnlineClient(prob) for prob in problem[:N_CLIENTS]]
        thread = threading.Thread(target=server.start)
        thread.start()
        time.sleep(1)
        try:
            # This code can only run without error in a non-test environment
            start_clients(clients)
        except ConnectionError:
            pass
        finally:
            server.shutdown()
            thread.join(timeout=1)

    def test_invalid_server_method(self):
        class InvalidServerHandler(Server):
            def handle_request(self, client_data: ClientPackage):
                raise RuntimeError('Test exception in server.handle_request()')

            def aggregate(self):
                pass

        class InvalidServerAgg(Server):
            def handle_request(self, client_data: ClientPackage):
                pass

            def aggregate(self):
                raise RuntimeError("Test raise error")

        for server_cls in [InvalidServerHandler, InvalidServerAgg]:
            with self.subTest(server_cls=server_cls):
                server = server_cls()
                server._add_client(1)
                thread = threading.Thread(target=server.start)
                thread.start()
                time.sleep(1)
                server.shutdown('Shutdown server thread in test')
                thread.join(timeout=1)
