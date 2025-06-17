import numpy as np
from collections import defaultdict
from typing import Callable
from pyfmto.framework import Server, ClientPackage, ServerPackage, Actions, DataArchive
from pyfmto.utilities import logger

from pyfmto.algorithms.TS import init_samples
from .fmtbo_utils import AggData


class FmtboServer(Server):

    def __init__(self):
        super().__init__()
        self.client_bounds = []
        self.clients_data = defaultdict(DataArchive)

        self.d_share_size = None
        self.d_share = None
        self.dim = None

    def create_d_share(self):
        if self.d_share is not None or len(self.client_bounds) < self.num_clients:
            return
        x_lb = None
        x_ub = None
        for new_lb, new_ub in self.client_bounds:
            x_lb, x_ub = self._set_bounds(x_lb, x_ub, new_lb, new_ub)
        self.d_share = init_samples(x_lb=x_lb,
                               x_ub=x_ub,
                               size=self.d_share_size,
                               dim=self.dim)
        logger.debug(f'd_share initialized, shape = {self.d_share.shape}')


    @staticmethod
    def _set_bounds(prev_lb: np.ndarray, prev_ub: np.ndarray, new_lb: np.ndarray, new_ub: np.ndarray):
        lb = new_lb if prev_lb is None else np.minimum(prev_lb, new_lb)
        ub = new_ub if prev_ub is None else np.maximum(prev_ub, new_ub)
        return lb, ub

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        action_map: dict[Actions, Callable[[ClientPackage], ServerPackage]] = {
            Actions.PUSH_INIT: self._return_save_status,
            Actions.PUSH_UPDATE: self._return_save_status,
            Actions.PULL_INIT: self._return_init_data,
            Actions.PULL_UPDATE: self._return_latest_update,
        }

        action = action_map.get(client_data.action)
        if action:
            return action(client_data)
        return ServerPackage('Unknown Action', data={'status': 'error'})

    def _return_save_status(self, client_data: ClientPackage) -> ServerPackage:
        if client_data.action == Actions.PUSH_INIT:
            self.dim = client_data.data['dim']
            self.d_share_size = client_data.data['d_share_size']
            self.client_bounds.append(client_data.data['bound'])
            self.create_d_share()
        else:
            self.clients_data[client_data.cid].add_src(client_data.data)
        return ServerPackage('SaveStatus', data={'status': 'ok'})

    def _return_init_data(self, client_data: ClientPackage) -> ServerPackage:
        if self.d_share is None:
            return ServerPackage('InitData', data=None)
        return ServerPackage('InitData', data={'d_share':self.d_share, 'theta': 5})

    def _return_latest_update(self, client_data: ClientPackage) -> ServerPackage:
        lts_upd: AggData = self.clients_data[client_data.cid].get_latest_res()
        return ServerPackage('LatestUpdate', data=lts_upd)

    def aggregate(self, client_id):
        src_num = np.sum(self.agg_src_versions > self.latest_res_version(client_id))
        if src_num == self.num_clients:
            index = self.clients_data[client_id].num_res
            ls_mat = self._cal_ls_mat_by_rank(index)
            for cid, ls_array in enumerate(ls_mat):
                selected_updates = self._select_updates_for_merge(ls_array)
                src_num = len(selected_updates)
                res = self._fedavg(selected_updates, index)
                ver = self.clients_data[cid+1].num_res + 1
                self.clients_data[cid+1].add_res(AggData(ver, src_num, res))
                logger.debug(f"Aggregated client {cid} ver {ver} result, shape={res.shape}")

    def _cal_ls_mat_by_rank(self, index):
        ls_mat = np.zeros((self.num_clients, self.num_clients))
        for cid in self.clients_data.keys():
            for _cid in self.clients_data.keys():
                if cid == _cid:
                    continue
                try:
                    update = self.clients_data[cid].src_data[index]
                    _update = self.clients_data[_cid].src_data[index]
                except IndexError:
                    print(f"Client id list {list(self.clients_data.keys())}")
                    print(f"Client {cid} source num: {self.clients_data[cid].num_src}")
                    print(f"Client {_cid} source num: {self.clients_data[_cid].num_src}")
                    continue
                rank, _rank = update['rank'], _update['rank']
                for r1, _r1 in zip(rank, _rank):
                    ls_mat[cid-1, _cid-1] += np.sum((r1 < rank) ^ (_r1 < _rank))
        return ls_mat

    def _select_updates_for_merge(self, ls_array):
        clients_number = int(len(ls_array) * 0.8)
        cid_list_of_sorted_array = np.argsort(ls_array)
        selected_cid = cid_list_of_sorted_array[0:clients_number]
        selected_updates = [self.clients_data[cid+1] for cid in selected_cid]
        return selected_updates

    @staticmethod
    def _fedavg(selected_updates: list[DataArchive], index):
        total_samples = 0
        gp_params = 0
        for update in selected_updates:
            update_data = update.src_data[index]
            total_samples += update_data['size']
            gp_params += update_data['size'] * update_data['global']
        gp_params /= total_samples
        return gp_params

    @property
    def agg_src_versions(self):
        src_versions = []
        for src_data in self.clients_data.values():
            src_versions.append(src_data.num_src)
        return np.array(src_versions)

    def latest_res_version(self, client_id):
        return self.clients_data[client_id].num_res