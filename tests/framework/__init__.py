import time
from datetime import date

from pyfmto.framework import Server, ClientPackage, ServerPackage, Client, record_runtime
from pyfmto.problems import SingleTaskProblem


class OfflineServer(Server):
    def __init__(self):
        """An offline server doesn't response any additional request"""
        super().__init__()
        self.alpha = 0.1
        self.beta = 0.2

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self):
        pass


class OnlineServer(Server):
    def __init__(self):
        super().__init__()
        self.set_agg_interval(0.5)
        self.update_server_info('time init', date.ctime(date.today()))
        self.update_server_info('time init', date.ctime(date.today()))
        self.update_server_info('num clients', str(self.num_clients))

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        return ServerPackage('response', 'server data')

    def aggregate(self):
        self.update_server_info('time stamp', date.ctime(date.today()))
        self.update_server_info('time stamp', date.ctime(date.today()))
        self.update_server_info('num clients', str(self.num_clients))


class OnlineClient(Client):
    def __init__(self, problem: SingleTaskProblem):
        """An online client will communicate with the server by the request() method"""
        super().__init__(problem)

    def optimize(self):
        time.sleep(0.5)
        self.push()
        self.optimizing()

    @record_runtime()  # cover the decorator and round info logging method
    def push(self):
        msg = "request failed"  # set a message to cover the message logging case
        pkg = ClientPackage(cid=self.id, action='test', data='client data')  # create a request package

        # Setting the request interval to 0.1 seconds to
        # speed up the process when requesting an offline server
        res = self.request_server(package=pkg, repeat=2, interval=0.1, msg=msg)
        if res:
            assert res.data == 'server data', f"res.data is {res.data} != 'server data'"

    @record_runtime('Optimizing')  # cover the custom record name
    def optimizing(self):
        x = self.problem.random_uniform_x(1)
        y = self.problem.evaluate(x)
        self.solutions.append(x, y)
