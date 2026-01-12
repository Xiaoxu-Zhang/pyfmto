from typing import Any

from pyfmto.framework import Client, Server, ClientPackage


class ClassPrefixClient(Client):
    """
    name: c
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.problem.auto_update_solutions = True

    def optimize(self):
        x = self.problem.random_uniform_x(1)
        self.problem.evaluate(x)
        self.request_server(package=ClientPackage(self.id, 'sync'), repeat=100)

    def check_pkg(self, pkg) -> bool:
        return pkg is not None


class ClassPrefixServer(Server):
    """
    name: s
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.all_joined = False

    def aggregate(self):
        if self.num_clients == 4:
            self.all_joined = True

    def handle_request(self, pkg: ClientPackage) -> Any:
        if self.all_joined:
            return 'OK'
        else:
            return None
