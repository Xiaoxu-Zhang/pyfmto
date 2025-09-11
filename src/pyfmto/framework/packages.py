from collections import defaultdict
from enum import Enum, auto
from typing import Optional, Any

__all__ = ['Actions', 'ClientPackage', 'ServerPackage', 'DataArchive', 'SyncDataManager']


class Actions(Enum):
    REGISTER = auto()
    QUIT = auto()


class ClientPackage:
    def __init__(self, cid: Optional[int], action: Any, data: Any = None):
        self.cid = cid
        self.action = action
        self.data = data


class ServerPackage:
    def __init__(self, desc: str, data=None):
        self.desc = desc
        self.data = data


class SyncDataManager:
    def __init__(self):
        self._source: dict[int, dict[int, Any]] = defaultdict(dict)
        self._result: dict[int, dict[int, Any]] = defaultdict(dict)

    def update_src(self, cid: int, version: int, data: Any):
        self._source[cid][version] = data

    def update_res(self, cid: int, version: int, data: Any):
        self._result[cid][version] = data

    def lts_src_ver(self, cid: int) -> int:
        return max(self._source.get(cid, {-1: None}).keys())

    def lts_res_ver(self, cid: int) -> int:
        return max(self._result.get(cid, {-1: None}).keys())

    def get_src(self, cid: int, version: int):
        try:
            return self._source[cid][version]
        except KeyError:
            return None

    def get_res(self, cid: int, version: int):
        try:
            return self._result[cid][version]
        except KeyError:
            return None

    @property
    def available_src_ver(self) -> int:
        vers = [max(data.keys()) for data in self._source.values()]
        if vers:
            return min(vers)
        else:
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
