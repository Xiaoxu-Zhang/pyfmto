import asyncio
import logging
import pickle
import time
import traceback
import uvicorn
import wrapt
from abc import abstractmethod, ABC
from collections import defaultdict
from fastapi import FastAPI, Response, Depends, Request
from setproctitle import setproctitle
from tabulate import tabulate
from typing import final, Optional

from .packages import ServerPackage, ClientPackage, Actions
from pyfmto.utilities import logger
app = FastAPI()


async def load_body(request: Request):
    raw_data = await request.body()
    if not raw_data:
        return None
    return pickle.loads(raw_data)


def catch_exception():
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        try:
            return wrapped(*args, **kwargs)
        except OSError as e:
            logger.error(f"{e}")
            instance.shutdown('OSError occurred')
        except Exception:
            traceback.print_exc()
            instance.shutdown('Exception occurred')
    return wrapper


class Server(ABC):
    def __init__(self):
        self._server: Optional[uvicorn.Server] = None
        self._active_clients = set()
        self._server_info = defaultdict(list)
        self._agg_interval = 0.5
        self._handler_pool_size = 2
        self._updated_server_info = False

        self._register_routes()
        self._config = uvicorn.Config(app)
        self.set_addr()
        self.set_log_print()

        self._quit = False
        self._idle_max = 0
        self._last_request_time = time.time()

    def set_addr(self, host='localhost', port=18510):
        self._config.host = host
        self._config.port = port

    def set_idle_max(self, idle_max: float):
        """
        Sets the maximum idle time for the server. If a positive value is provided,
        the server will automatically shut down after being idle for longer than this duration.

        In some cases, due to client implementation limitations, pressing `Ctrl+C` may fail to
        terminate the server process properly. By setting an appropriate idle timeout, you can
        ensure that the server exits gracefully after prolonged inactivity.

        If set to zero or a negative value, automatic termination is disabled and the server
        will not shut down due to idleness.

        Parameters
        ----------
        idle_max : float
            Maximum idle time in seconds. It is recommended to set this value slightly greater
            than the maximum interval between client requests to ensure stability and correctness.
        """
        self._idle_max = max(0.0, idle_max)

    @staticmethod
    def set_log_print(yes=False):
        level = logging.INFO if yes else logging.ERROR
        disabled = not yes

        for name in ["uvicorn", "fastapi", "uvicorn.error", "uvicorn.access"]:
            __logger = logging.getLogger(name)
            __logger.setLevel(level)
            __logger.disabled = disabled

    def set_agg_interval(self, seconds: float):
        self._agg_interval = max(0.01, seconds)

    def update_server_info(self, name: str, value: str):
        if len(self._server_info[name]) == 1:
            if self._server_info[name][0] != value:
                self._server_info[name][0] = value
                self._updated_server_info = True
        else:
            self._server_info[name] = [value]
            self._updated_server_info = True

    def start(self):
        setproctitle(f'AlgServer')
        self._server = uvicorn.Server(self._config)
        asyncio.run(self._run_server())

    async def _run_server(self):
        await asyncio.gather(
            self._server.serve(),
            self._monitor(),
            self._aggregator()
        )

    @catch_exception()
    async def _aggregator(self):
        while not self._quit:
            await asyncio.sleep(self._agg_interval)
            peers_id = self.sorted_ids
            logger.debug(f"Server aggregating {self.num_clients} clients data")
            for cid in peers_id:
                self.aggregate(cid)

    def _register_routes(self):
        @app.post("/alg-comm")
        async def alg_comm(client_pkg: ClientPackage = Depends(load_body)):
            self._last_request_time = time.time()
            server_pkg = self._handle_request(client_pkg)
            return Response(content=pickle.dumps(server_pkg), media_type="application/x-pickle")

    async def _monitor(self):
        await asyncio.sleep(10)
        while not self._quit:
            self._log_server_info()
            self._idle_quit()
            await asyncio.sleep(3)

    @catch_exception()
    def _log_server_info(self):
        if self._updated_server_info and self._server_info is not None:
            tab = tabulate(self._server_info, headers="keys", tablefmt="psql")
            logger.info(f"\n{'=' * 30} Saved {len(self._server_info)} clients data {'=' * 30}\n{tab}")
            self._updated_server_info = False

    def _idle_quit(self):
        if self._idle_max > 0:
            idle_len = time.time() - self._last_request_time
            if idle_len > self._idle_max:
                self.shutdown(f"Idle for {idle_len:.2f} seconds")

    @final
    @catch_exception()
    def _handle_request(self, data: ClientPackage) -> ServerPackage:
        if not data:
            return ServerPackage(desc='error', data='request without data')
        elif data.action == Actions.REGISTER:
            self._add_client(data.cid)
            return ServerPackage(desc='register', data='join success')
        elif data.action == Actions.QUIT:
            self._del_client(data.cid)
            return ServerPackage(desc='quit', data='quit success')
        else:
            return self.handle_request(data)

    @abstractmethod
    def handle_request(self, client_data: ClientPackage) -> ServerPackage: ...

    @abstractmethod
    def aggregate(self, client_id): ...

    @staticmethod
    def named_client(cid):
        return f"Client {cid:<2}"

    def _add_client(self, client_id):
        self._active_clients.add(client_id)
        logger.info(f"Client {client_id} join, total {self.num_clients} clients")

    def _del_client(self, client_id):
        self._active_clients.remove(client_id)
        logger.info(f"Client {client_id} quit, remain {self.num_clients} clients")
        if self.num_clients == 0:
            self.shutdown('No active clients')

    def shutdown(self, msg='no message'):
        if not self._quit:
            logger.info(f"Server shutting down ({msg})")
            self._quit = True
            self._server.should_exit = True

    @property
    def sorted_ids(self):
        return sorted(self._active_clients)

    @property
    def num_clients(self):
        return len(self._active_clients)

    @property
    def started(self):
        return self._server.started
