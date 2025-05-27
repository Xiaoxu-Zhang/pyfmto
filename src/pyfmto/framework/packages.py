
from enum import Enum, auto
from typing import Optional, Any

__all__ = ['Actions', 'ClientPackage', 'ServerPackage', 'DataArchive']

class Actions(Enum):
    REGISTER = auto()
    PUSH_INIT = auto()
    PULL_INIT = auto()
    PULL_UPDATE = auto()
    PUSH_UPDATE = auto()
    QUIT = auto()


class ClientPackage:
    def __init__(self, cid: Optional[int], action: Any, data: Any=None):
        self.cid = cid
        self.action = action
        self.data = data


class ServerPackage:
    def __init__(self, desc: str, data=None):
        self.desc = desc
        self.data = data


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
