from typing import Any

from ..core import ComponentData
from .client import Client, record_runtime
from .packages import ClientPackage, SyncDataManager
from .server import Server

__all__ = [
    'AlgorithmData',
    'Client',
    'ClientPackage',
    'Server',
    'SyncDataManager',
    'record_runtime'
]


class AlgorithmData(ComponentData):
    client: type[Client]
    server: type[Server]

    def __init__(self):
        super().__init__()

    @property
    def available(self):
        clt_ok = self._check_attr('client', Client)
        srv_ok = self._check_attr('server', Server)
        return clt_ok and srv_ok

    def load_default_params(self) -> None:
        clt_params = self._parse_default_params('client')
        srv_params = self._parse_default_params('server')
        self.params_default.update(
            client=clt_params,
            server=srv_params
        )

    @property
    def params(self) -> dict[str, Any]:
        return self._merged_params()
