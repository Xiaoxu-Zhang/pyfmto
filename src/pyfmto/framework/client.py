import pickle
import requests
import time
import traceback
import wrapt
from abc import abstractmethod, ABC
from collections import defaultdict
from numpy import ndarray
from requests.exceptions import ConnectionError
from tqdm import tqdm
from typing import final, Optional, Any
from yaml import safe_load, MarkedYAMLError

from .packages import ClientPackage, ServerPackage, Actions
from pyfmto.problems import SingleTaskProblem
from pyfmto.utilities import logger, titled_tabulate, tabulate_formats as tf

__all__ = [
    'Client',
    'record_runtime'
]


def record_runtime(name=None):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        start_time = time.time()
        result = wrapped(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        if name is None:
            n = wrapped.__name__
        else:
            n = name
        instance.record_round_info(n, f"[{runtime:.2f}s]")
        return result

    return wrapper


class Client(ABC):
    def __init__(self, problem: SingleTaskProblem):
        self._url = None
        self._conn_retry = None
        self.problem = problem
        self._round_info = defaultdict(list)

        self.set_addr()

    def set_addr(self, ip='localhost', port=18510, conn_retry: int=3):
        """
        Configure the server address and port that this client will connect to.

        Parameters:
        -----------
        ip : str, optional
            The IP address of the server. Default is 'localhost'.

        port : int, optional
            The port number on which the server is listening. Default is 18510.

        conn_retry : int, optional
            Maximum number of connection retry attempts if the initial request fails.
            This value determines how many times the client will attempt to reconnect
            before raising an exception. Default is 3.

        Returns:
        --------
        None
            This method does not return any value.
        """
        self._url = f"http://{ip}:{port}"
        self._conn_retry = conn_retry

    def __logging_params(self):
        param_dict = defaultdict(list)
        param_dict['TaskName'].append(self.problem.name)
        param_dict['Dim'].append(self.problem.dim)
        param_dict['Obj'].append(self.problem.obj)
        param_dict['IID'].append(self.problem.np_per_dim)
        param_dict['IniFE'].append(self.problem.fe_init)
        param_dict['MaxFE'].append(self.problem.fe_max)
        tab = titled_tabulate(
            f"{self.name} Params", '=',
            param_dict, headers='keys', tablefmt=tf.rounded_grid
        )
        logger.info(tab)

    def __logging_round_info(self, only_latest: bool):
        if only_latest:
            data = {}
            for k, v in self._round_info.items():
                data[k] = v[-1:]
        else:
            data = self._round_info
        if self._round_info:
            tab = titled_tabulate(
                f"{self.name} {self.problem.name}({self.dim}D)",
                '=', data, headers='keys', tablefmt=tf.rounded_grid
            )
            logger.info(tab)

    def record_round_info(self, name:str, value: str):
        self._round_info[name].append(value)

    @final
    def start(self):
        try:
            logger.info(f"{self.name} started")
            self.__register_id()
            self.__logging_params()

            pbar = tqdm(
                total=self.fe_max,
                initial=self.solutions.num_updated,
                desc=self.name, unit="Round",
                ncols=100, position=self.id, leave=False)

            while self.problem.fe_available > 0:
                self.optimize()
                pbar.update(self.solutions.num_updated)
                self.__logging_round_info(only_latest=True)

            self.__logging_round_info(only_latest=False)
            self.send_quit()
            logger.info(f"{self.name} exit with available FE = {self.problem.fe_available}")
        except Exception:
            self.send_quit()
            if self.id == 1:
                print(f"Traceback of {self.name}")
                traceback.print_exc()
            logger.info(f"{self.name} exit with available FE = {self.problem.fe_available}")
            exit(-1)
        return self.id, self.solutions

    @abstractmethod
    def optimize(self): ...

    def __register_id(self):
        while True:
            pkg = ClientPackage(self.id, Actions.REGISTER)
            res = self.request_server(pkg)
            if res is not None:
                logger.debug(f"{self.name} registered")
                break

    @staticmethod
    def deserialize_pickle(package: Any):
        return pickle.loads(package) if package is not None else None

    def request_server(self, package: Any,
                       repeat: int=10, interval: float=1.,
                       msg=None) -> Optional[ServerPackage]:
        """
        Send a request to the server and wait for a response that satisfies a given condition.

        Parameters
        ----------
        package : Any
            The data package to send to the server.
        repeat : int, optional
            The maximum number of attempts to receive a valid response. Default is 10.
        interval : float, optional
            The time interval (in seconds) between attempts. Default is 1 second. Note
            that even the repeat is 1, the interval still effective.
        msg : str, optional
            Message that output to log after every failed repeat using debug level.

        Returns
        -------
        Any
            The response from the server if an acceptable response is received within
            the specified number of attempts.
            Returns None if no acceptable response is received.

        Notes
        -----
        This method repeatedly sends a request to the server and waits for a response.
        It will continue to attempt to receive a response up to `repeat` times, with a
        delay of `interval` seconds between each attempt.

        The response is considered acceptable if it passes the `check_pkg` method, which
        determines whether the received response is valid based on personalized criteria.
        If the response does not meet these criteria, the method will perform the next repeat.
        """
        repeat_max = max(1, repeat)
        curr_repeat = 1
        conn_retry = 0
        while curr_repeat <= repeat_max:
            if msg:
                logger.debug(f"{self.name} [Request retry {curr_repeat}/{repeat_max}] {msg}")
            data = pickle.dumps(package)
            try:
                res = requests.post(f"{self._url}/alg-comm", data=data, headers={"Content-Type": "application/x-pickle"})
                if res.status_code == 200:
                    server_pkg = self.deserialize_pickle(res.content)
                    if self.check_pkg(server_pkg):
                        return server_pkg
            except ConnectionError:
                time.sleep(interval)
                conn_retry += 1
                logger.error(f"{self.name} Connection refused {conn_retry} times.")
                if conn_retry >= self._conn_retry:
                    raise ConnectionError(f"{self.name} Connection failed {conn_retry} times.")
                continue
            curr_repeat += 1
            time.sleep(interval)

    def check_pkg(self, x) -> bool:
        """
        Determine whether the response is acceptable by check the specific data within it.

        Parameters
        ----------
        x : Any
            The response received from the server. It is guaranteed to be non-None.

        Returns
        -------
        bool
            True if the data within the response is acceptable; otherwise, False.

        Notes
        -----
        This method allow additional validation for the response data. By default, it
        does not perform any specific checks and returns `True` for any non-None response.

        Subclasses can override this method to implement custom validation logic. Refer
        to the `EXAMPLE` algorithm for a detailed implementation.
        """
        return x is not None

    def send_quit(self):
        quit_pkg = ClientPackage(self.id, Actions.QUIT)
        self.request_server(quit_pkg)

    def load_default_kwargs(self):
        try:
            return safe_load(self.__class__.__doc__)
        except MarkedYAMLError:
            raise

    @property
    def id(self):
        """Same with the problem id"""
        return self.problem.id

    @property
    def name(self):
        return f"Client {self.id:<2}"

    @property
    def dim(self) -> int:
        return self.problem.dim

    @property
    def obj(self) -> int:
        return self.problem.obj

    @property
    def fe_init(self) -> int:
        return self.problem.fe_init

    @property
    def fe_max(self) -> int:
        return self.problem.fe_max

    @property
    def y_max(self) -> float:
        return self.solutions.y_max

    @property
    def y_min(self) -> float:
        return self.solutions.y_min

    @property
    def x_lb(self) -> ndarray:
        return self.problem.x_lb

    @property
    def x_ub(self) -> ndarray:
        return self.problem.x_ub

    @property
    def solutions(self):
        return self.problem.solutions
