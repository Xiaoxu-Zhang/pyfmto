import asyncio
import logging
import pickle
import time
import traceback
import uvicorn
from abc import abstractmethod, ABC
from collections import defaultdict
from fastapi import FastAPI, Response, Depends, Request
from setproctitle import setproctitle
from tabulate import tabulate
from typing import final, Any

from .packages import ClientPackage, Actions
from pyfmto.utilities import logger, parse_yaml
from ..utilities.tools import update_kwargs

app = FastAPI()


async def load_body(request: Request):
    raw_data = await request.body()
    if not raw_data:
        return None
    return pickle.loads(raw_data)


class Server(ABC):
    _server: uvicorn.Server

    def __init__(self):
        self._active_clients = set()
        self._server_info = defaultdict(list)
        self._agg_interval = 0.5
        self._handler_pool_size = 2
        self._updated_server_info = False

        self._register_routes()
        self._config = uvicorn.Config(app)
        self.set_addr()
        self._disable_consol_log()

        self._quit = False
        self._last_request_time = time.time()

    def set_addr(self, host='localhost', port=18510):
        self._config.host = host
        self._config.port = port

    @staticmethod
    def enable_consol_log():
        for name in ["uvicorn", "fastapi", "uvicorn.error", "uvicorn.access"]:
            __logger = logging.getLogger(name)
            __logger.setLevel(logging.INFO)
            __logger.disabled = False

    @staticmethod
    def _disable_consol_log():
        for name in ["uvicorn", "fastapi", "uvicorn.error", "uvicorn.access"]:
            __logger = logging.getLogger(name)
            __logger.setLevel(logging.ERROR)
            __logger.disabled = True

    def set_agg_interval(self, seconds: float):
        self._agg_interval = max(0.01, seconds)

    def update_server_info(self, name: str, value: str):
        lts = self._server_info[name][-1:]
        if [value] == lts:
            return
        else:
            self._server_info[name].append(value)
            self._updated_server_info = True

    def start(self):
        setproctitle('AlgServer')
        self._server = uvicorn.Server(self._config)
        asyncio.run(self._run_server())

    async def _run_server(self):
        await asyncio.gather(
            self._server.serve(),
            self._monitor(),
            self._aggregator()
        )

    async def _aggregator(self):
        while not self._quit:
            await asyncio.sleep(self._agg_interval)
            if self.num_clients == 0:
                continue
            try:
                self.aggregate()
            except Exception:
                logger.error(f"Server error: {traceback.format_exc()}")
                self.shutdown('aggregate error')

    async def _monitor(self):
        await asyncio.sleep(1)
        while not self._quit:
            self._log_server_info()
            await asyncio.sleep(.5)

    def _register_routes(self):
        @app.post("/alg-comm")
        async def alg_comm(client_pkg: ClientPackage = Depends(load_body)):
            self._last_request_time = time.time()
            try:
                server_pkg = self._handle_request(client_pkg)
            except Exception:
                logger.error(traceback.format_exc())
                self.shutdown(traceback.format_exc())
                server_pkg = None
            return Response(content=pickle.dumps(server_pkg), media_type="application/x-pickle")

    def _log_server_info(self):
        data: dict[str, list[str]] = {}
        for key, value in self._server_info.items():
            if len(value) > 0:
                data[key] = value[-1:]
        if data and self._updated_server_info:
            tab = tabulate(data, headers="keys", tablefmt="psql")
            logger.info(f"\n{'=' * 30} Saved {len(self._server_info)} clients data {'=' * 30}\n{tab}")
            self._updated_server_info = False

    @final
    def _handle_request(self, data: ClientPackage):
        if data.action == Actions.REGISTER:
            self._add_client(data.cid)
            return 'join success'
        elif data.action == Actions.QUIT:
            self._del_client(data.cid)
            return 'quit success'
        else:
            return self.handle_request(data)

    @abstractmethod
    def handle_request(self, client_data: ClientPackage) -> Any:
        ...  # pragma: no cover

    @abstractmethod
    def aggregate(self):
        ...  # pragma: no cover

    def _add_client(self, client_id):
        self._active_clients.add(client_id)
        logger.info(f"Client {client_id} join, total {self.num_clients} clients")

    def _del_client(self, client_id):
        self._active_clients.remove(client_id)
        logger.info(f"Client {client_id} quit, remain {self.num_clients} clients")
        if self.num_clients < 1:
            self.shutdown('No active clients')

    def shutdown(self, msg='no message'):
        if not self._quit:
            logger.info(f"Server shutting down ({msg})")
            self._quit = True
            self._server.should_exit = True
            self._server.force_exit = True

    def update_kwargs(self, kwargs: dict):
        docstr = parse_yaml(self.__class__.__doc__)
        return update_kwargs(self.__class__.__name__, docstr, kwargs)

    @property
    def sorted_ids(self):
        return sorted(self._active_clients)

    @property
    def num_clients(self):
        return len(self._active_clients)
