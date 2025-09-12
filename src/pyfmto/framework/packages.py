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


class ServerPackage:
    def __init__(self, desc: str, data=None):
        self.desc = desc
        self.data = data


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
        try:
            data = self._source[cid]
            return max(data.keys())
        except (ValueError, KeyError):
            return -1

    @validate_call
    def lts_res_ver(self, cid: int) -> int:
        try:
            data = self._result[cid]
            return max(data.keys())
        except (ValueError, KeyError):
            return -1

    @validate_call
    def get_src(self, cid: int, version: int):
        try:
            return self._source[cid][version]
        except KeyError:
            logger.debug(f"cid={cid} version={version} not found in source")
            return None

    @validate_call
    def get_res(self, cid: int, version: int):
        try:
            return self._result[cid][version]
        except KeyError:
            if cid not in self._result:
                logger.debug(f"Client id '0' not in source data")
            else:
                logger.debug(f"Client id '{cid}' source data version={version} not found")
            return None

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
