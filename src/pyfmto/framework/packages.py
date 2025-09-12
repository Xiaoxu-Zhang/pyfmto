from collections import defaultdict
from enum import Enum, auto
from typing import Optional, Any

__all__ = ['Actions', 'ClientPackage', 'DataArchive', 'SyncDataManager']

from pydantic import validate_call

from pyfmto.utilities import logger


class Actions(Enum):
    REGISTER = auto()
    QUIT = auto()


class ClientPackage:
    def __init__(self, cid: Optional[int], action: Any):
        self.cid = cid
        self.action = action


class SyncDataManager:
    def __init__(self):
        self._source: dict[int, dict[int, Any]] = defaultdict(dict)
        self._result: dict[int, dict[int, Any]] = defaultdict(dict)

    @validate_call
    def update_src(self, cid: int, version: int, data: Any):
        self._source[cid][version] = data

    @validate_call
    def update_res(self, cid: int, version: int, data: Any):
        self._result[cid][version] = data

    @validate_call
    def lts_src_ver(self, cid: int) -> int:
        data = self._source.get(cid, {-1: None})
        return max(data.keys())

    @validate_call
    def lts_res_ver(self, cid: int) -> int:
        data = self._result.get(cid, {-1: None})
        return max(data.keys())

    @validate_call
    def get_src(self, cid: int, version: int):
        if cid not in self._source:
            logger.debug(f"cid={cid} not found in source")
            return None
        data = self._source[cid].get(version)
        if data is None:
            logger.debug(f"cid={cid} version={version} not found in source")
        return data

    @validate_call
    def get_res(self, cid: int, version: int):
        if cid not in self._result:
            logger.debug(f"cid={cid} not found in result")
            return None
        data = self._result[cid].get(version)
        if data is None:
            logger.debug(f"cid={cid} version={version} not found in result")
        return data

    @property
    def available_src_ver(self) -> int:
        try:
            return min([max(data.keys()) for data in self._source.values()])
        except ValueError:
            return -1

    @property
    def num_clients(self) -> int:
        return len(self._source)


class DataArchive:
    def __init__(self):
        self.src_data = []
        self.res_data = []

    @property
    def num_src(self) -> int:
        return len(self.src_data)

    @property
    def num_res(self) -> int:
        return len(self.res_data)

    def add_src(self, src_data):
        self.src_data.append(src_data)

    def add_res(self, agg_data):
        self.res_data.append(agg_data)

    def get_latest_res(self):
        return self.res_data[-1] if self.num_res > 0 else None

    def get_latest_src(self):
        return self.src_data[-1] if self.num_src > 0 else None
