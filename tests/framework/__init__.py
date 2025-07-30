import time

from pyfmto.framework import Server, ClientPackage, ServerPackage, Client, record_runtime
from pyfmto.problems import SingleTaskProblem


class OfflineServer(Server):
    def __init__(self):
        """An offline server doesn't response any additional request"""
        super().__init__()

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass


class OnlineServer(Server):
    def __init__(self):
        super().__init__()

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        return ServerPackage('response', 'server data')

    def aggregate(self, client_id):
        pass


class InvalidServer(Server):
    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        raise RuntimeError("Test raise error")


class EmptyClient(Client):
    def __init__(self, problem: SingleTaskProblem):
        super().__init__(problem)

    def optimize(self):
        pass


class ConfigurableClient(Client):
    """
    alpha: 0.1
    beta: 0.8
    gamma: 0.5
    phi: 0.1
    theta: 0.0
    """
    def __init__(self, problem: SingleTaskProblem, **kwargs):
        super().__init__(problem)
        kwargs = self.update_kwargs(kwargs)
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.phi = kwargs['phi']
        self.theta = kwargs['theta']

    def optimize(self):
        pass


class OfflineClient(Client):
    def __init__(self, problem: SingleTaskProblem):
        """An offline client doesn't communicate with the server"""
        super().__init__(problem)

    def optimize(self):
        time.sleep(0.1)
        x = self.problem.random_uniform_x(1)
        y = self.problem.evaluate(x)
        self.problem.solutions.append(x, y)


class OnlineClient(Client):
    def __init__(self, problem: SingleTaskProblem):
        """An online client will communicate with the server by the request() method"""
        super().__init__(problem)

    def optimize(self):
        time.sleep(0.1) # sleep for 0.1 seconds to avoid frequent requests
        self.push()
        self.optimizing()

    @record_runtime() # cover the decorator and round info logging method
    def push(self):
        msg = f"request failed" # set a message to cover the message logging case
        pkg = ClientPackage(cid=self.id, action='test', data='client data') # create a request package

        # Setting the request interval to 0.1 seconds to
        # speed up the process when requesting an offline server
        res = self.request_server(package=pkg, repeat=2, interval=0.1, msg=msg)
        if res:
            assert res.data == 'server data', f"res.data is {res.data} != 'server data'"

    @record_runtime('Optimizing') # cover the custom record name
    def optimizing(self):
        x = self.problem.random_uniform_x(1)
        y = self.problem.evaluate(x)
        self.solutions.append(x, y)


class InvalidClient(Client):
    def __init__(self, problem: SingleTaskProblem):
        super().__init__(problem)

    def optimize(self):
        raise RuntimeError("Test raise error")
