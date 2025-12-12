import pickle
import requests  # type: ignore
import time
import traceback
import wrapt
from abc import abstractmethod, ABC
from collections import defaultdict
from numpy import ndarray
from requests.exceptions import ConnectionError  # type: ignore
from rich.progress import Progress, TaskID
from typing import final, Any

from .packages import ClientPackage, Actions
from pyfmto.problems import SingleTaskProblem
from pyfmto.utilities import logger, colored, titled_tabulate, tabulate_formats as tf

__all__ = [
    'Client',
    'record_runtime'
]


def record_runtime(name=None):
    @wrapt.decorator  # type: ignore  # wrapt.decorator lacks types in Python 3.9
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
    progress: Progress
    bar: TaskID

    def __init__(self, problem: SingleTaskProblem, **kwargs):
        self._url: str = ''
        self.conn_retry: int = 3
        self.problem = problem
        self.rounds_info: dict[str, list[str]] = defaultdict(list)

        self.set_addr()

    def set_addr(self, ip='localhost', port=18510) -> None:
        """
        Configure the server address and port that this client will connect to.

        Parameters:
        -----------
        ip : str, optional
            The IP address of the server. Default is 'localhost'.

        port : int, optional
            The port number on which the server is listening. Default is 18510.

        Returns:
        --------
        None
            This method does not return any value.
        """
        # noinspection HttpUrlsUsage
        self._url = f"http://{ip}:{port}"

    def __logging_start_info(self):
        param_dict = defaultdict(list)
        param_dict['TaskName'].append(self.problem.name)
        param_dict['Dim'].append(str(self.problem.dim))
        param_dict['Obj'].append(str(self.problem.obj))
        param_dict['IID'].append(str(self.problem.npd))
        param_dict['IniFE'].append(str(self.problem.fe_init))
        param_dict['MaxFE'].append(str(self.problem.fe_max))
        tab = titled_tabulate(
            f"{self.name} init information", '=',
            param_dict, headers='keys', tablefmt=tf.rounded_grid
        )
        logger.debug(tab)

    def record_round_info(self, name: str, value: str):
        self.rounds_info[name].append(value)

    @final
    def start(self):
        try:
            logger.info(f"{self.name} started")
            self.__register_id()
            self.__logging_start_info()

            self.__refresh_progress()
            while self.problem.fe_available > 0:
                self.optimize()
                self.__refresh_progress()

            self.send_quit()
            logger.debug(f"{self.name} exit with available FE = {self.problem.fe_available}")
        except Exception:
            self.send_quit()
            if self.id == 1:
                logger.error(traceback.format_exc())
                print(f"Error occurred while in {colored('self.optimize()', 'red')}, "
                      f"see {colored('out/logs/pyfmto.log', 'green')} for detail.")
            logger.info(f"{self.name} exit with available FE = {self.problem.fe_available}")
            raise
        return self

    def __refresh_progress(self):
        if getattr(self, 'progress', None) is None or getattr(self, 'bar', None) is None:
            logger.debug(f"{self.name} Rich's progress or TaskID object is None")
        else:
            self.progress.advance(self.bar, self.solutions.num_updated)

    @abstractmethod
    def optimize(self):
        ...  # pragma: no cover

    def __register_id(self):
        pkg = ClientPackage(self.id, Actions.REGISTER)
        self.request_server(pkg)
        logger.debug(f"{self.name} registered")

    @staticmethod
    def __uppack(package: Any):
        return pickle.loads(package) if package is not None else None

    def request_server(self, package: ClientPackage,
                       repeat: int = 10, interval: float = 1.,
                       msg=None) -> Any:
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
            Return Any, if all request attempts failure, return None

        Notes
        -----
        This method repeatedly sends a request to the server and waits for a response.
        It will continue to attempt to receive a response up to `repeat` times, with a
        delay of `interval` seconds between each attempt.

        The response is considered acceptable if it passes the `check_pkg` method, which
        determines whether the received response is valid based on personalized criteria.
        If the response does not meet these criteria, the method will perform the next repeat.
        """
        if not isinstance(package, ClientPackage):
            raise ValueError("package should be ClientPackage")
        repeat_max = max(1, repeat)
        request_repeat = 1
        failed_retry = 1
        while request_repeat <= repeat_max:
            if msg:
                logger.debug(f"{self.name} [Request retry {request_repeat}/{repeat_max}] {msg}")
            data = pickle.dumps(package)
            try:
                resp = requests.post(
                    url=f"{self._url}/alg-comm",
                    data=data,
                    headers={"Content-Type": "application/x-pickle"}
                )
            except ConnectionError:
                logger.error(f"{self.name} Connection failed {failed_retry} times.")
                time.sleep(interval)
                failed_retry += 1
                if failed_retry > self.conn_retry:
                    raise ConnectionError(f"{self.name} Connection failed {self.conn_retry} times.")
                continue
            pkg = self.__uppack(resp.content)
            if pkg is not None and self.check_pkg(pkg):
                return pkg
            else:
                time.sleep(interval)
                request_repeat += 1
        return None

    def check_pkg(self, pkg) -> bool:
        """
        Determine whether the response is acceptable by check the specific data within it.

        Parameters
        ----------
        pkg :
            The response received from the server.

        Returns
        -------
        bool
            True(default) if the package is acceptable; otherwise, False.

        Notes
        -----
        This method allow additional validation for the response data. By default, it
        does not perform any specific checks and returns `True` for any non-None response.

        Subclasses can override this method to implement custom validation logic. Refer
        to the `EXAMPLE` algorithm for a detailed implementation.
        """
        return True

    def send_quit(self):
        quit_pkg = ClientPackage(self.id, Actions.QUIT)
        self.request_server(quit_pkg)

    @property
    def id(self) -> int:
        """Using the problem id as client's id"""
        return self.problem.id

    @property
    def name(self) -> str:
        return f"Client {self.id:<2}"

    @property
    def dim(self) -> int:
        """The problem dimensions of the client"""
        return self.problem.dim

    @property
    def obj(self) -> int:
        """The problem's objectives of the client"""
        return self.problem.obj

    @property
    def fe_init(self) -> int:
        """The problem's initial random fitness evaluations of the client"""
        return self.problem.fe_init

    @property
    def fe_max(self) -> int:
        """The problem's maximum fitness evaluations of the client"""
        return self.problem.fe_max

    @property
    def y_max(self) -> float:
        """The problem's current maximum objective value of the client"""
        return self.solutions.y_max

    @property
    def y_min(self) -> float:
        """The problem's current minimum objective value of the client"""
        return self.solutions.y_min

    @property
    def lb(self) -> ndarray:
        """The problem's lower bounds of the client"""
        return self.problem.lb

    @property
    def ub(self) -> ndarray:
        """The problem's upper bounds of the client"""
        return self.problem.ub

    @property
    def solutions(self):
        """The problem's solutions object of the client"""
        return self.problem.solutions
