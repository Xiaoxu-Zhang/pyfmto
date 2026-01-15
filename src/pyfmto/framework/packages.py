from collections import defaultdict
from enum import Enum, auto
from typing import Any, Optional

from pydantic import validate_call

from ..utilities.loggers import logger
from ..utilities.tools import tabulate_formats, titled_tabulate

__all__ = ['Actions', 'ClientPackage', 'SyncDataManager']


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
    def update_src(self, cid: int, version: int, data: Any) -> None:
        """
        Update aggregation source of a client
        """
        self._source[cid][version] = data

    @validate_call
    def update_res(self, cid: int, version: int, data: Any) -> None:
        """
        Update aggregation results of a client
        """
        self._result[cid][version] = data

    @validate_call
    def lts_src_ver(self, cid: int) -> int:
        """
        Return the latest source version of a client
        """
        data = self._source.get(cid, {-1: None})
        return max(data.keys())

    @validate_call
    def lts_res_ver(self, cid: int) -> int:
        """
        Return the latest aggregation result version of a client
        """
        data = self._result.get(cid, {-1: None})
        return max(data.keys())

    @validate_call
    def get_src(self, cid: int, version: int) -> Any:
        """
        - Return the aggregation source
        - Return None if not found
        """
        if cid not in self._source:
            logger.debug(f"cid={cid} not found in source")
            return None
        data = self._source[cid].get(version)
        if data is None:
            logger.debug(f"cid={cid} version={version} not found in source")
        return data

    @validate_call
    def get_res(self, cid: int, version: int) -> Any:
        """
        - Return the aggregation result data
        - Return None if not found
        """
        if cid not in self._result:
            logger.debug(f"cid={cid} not found in result")
            return None
        data = self._result[cid].get(version)
        if data is None:
            logger.debug(f"cid={cid} version={version} not found in result")
        return data

    @property
    def available_src_ver(self) -> int:
        """
        - The latest consistent version uploaded by all clients
        - -1 if no source data is available
        """
        try:
            return min([max(data.keys()) for data in self._source.values()])
        except ValueError:
            return -1

    @property
    def num_clients(self) -> int:
        """
        The number of clients that have uploaded least 1 source data
        """
        return len(self._source)

    @property
    def data_info(self):
        src = {f"C {cid}": [self.lts_src_ver(cid)] for cid in sorted(self._source.keys())}
        res = {f"C {cid}": [self.lts_res_ver(cid)] for cid in sorted(self._result.keys())}
        src_str = titled_tabulate("Aggregation Source", '=',
                                  src, headers='keys', tablefmt=tabulate_formats.rounded_grid) if src else ""
        res_str = titled_tabulate("Aggregation Results", '=',
                                  res, headers='keys', tablefmt=tabulate_formats.rounded_grid) if res else ""
        return (
            f"AvaSrcVer: {self.available_src_ver}\n"
            f"Total Clt: {self.num_clients}"
            f"{src_str}{res_str}"
        )
