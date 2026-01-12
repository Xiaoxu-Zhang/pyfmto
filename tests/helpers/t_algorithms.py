from typing import Any

from pyfmto.framework import Client, Server, ClientPackage


class ClassPrefixClient(Client):
    """
    name: c
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.problem.auto_update_solutions = True
        self.conn_retry = 100

    def optimize(self):
        x = self.problem.random_uniform_x(1)
        self.problem.evaluate(x)


class ClassPrefixServer(Server):
    """
    name: s
    """

    def __init__(self, **kwargs):
        super().__init__()

    def aggregate(self):
        pass

    def handle_request(self, pkg: ClientPackage) -> Any:
        pass
