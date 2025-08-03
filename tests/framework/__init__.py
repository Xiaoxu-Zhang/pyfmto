import subprocess
import sys
import time
from typing import Type

import numpy as np

from pyfmto.framework import Server, ClientPackage, ServerPackage, Client, record_runtime
from pyfmto.problems import SingleTaskProblem


def start_subprocess_clients(client: Type[Client]):
    module_name = client.__module__
    class_name = client.__name__
    cmd = [
        sys.executable, "-c",
        f"from {module_name} import {class_name}; "
        f"from pyfmto.problems import load_problem; "
        f"from pyfmto.experiments.utils import start_clients; "
        f"prob = load_problem('tetci2019', fe_init=20, fe_max=25); "
        f"clients = [{class_name}(p) for p in prob[:3]]; "
        f"start_clients(clients)"
    ]
    print('\n'.join(cmd[2].split('; ')))
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


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

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        return ServerPackage('response', 'server data')

    def aggregate(self):
        rand = np.random.random()
        time.sleep(rand)
        self.update_server_info('round sleep', f"sleep {rand} s")


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
