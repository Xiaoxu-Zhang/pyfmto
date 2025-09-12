import numpy as np
from typing import Callable, Any
from pyfmto.framework import Server, SyncDataManager
from pyfmto.utilities import logger

from .fmtbo_utils import init_samples, Actions, ClientPackage


class FmtboServer(Server):
    """
    d_share_size: 50
    agg_proportion: 0.8
    """
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = self.update_kwargs(kwargs)
        self.client_bounds = []
        self.clients_data = SyncDataManager()
        self.done_vers: set[int] = {-1}
        self.d_share_size = kwargs['d_share_size']
        self.agg_proportion = kwargs['agg_proportion']
        self.d_share = None
        self.dim = None

    def create_d_share(self):
        if self.d_share is not None or len(self.client_bounds) < self.num_clients:
            return
        x_lb = None
        x_ub = None
        for new_lb, new_ub in self.client_bounds:
            x_lb, x_ub = self._set_bounds(x_lb, x_ub, new_lb, new_ub)
        self.d_share = init_samples(lb=x_lb, ub=x_ub, n_samples=self.d_share_size, dim=self.dim)
        logger.debug(f'd_share initialized, shape = {self.d_share.shape}')

    @staticmethod
    def _set_bounds(prev_lb: np.ndarray, prev_ub: np.ndarray, new_lb: np.ndarray, new_ub: np.ndarray):
        lb = new_lb if prev_lb is None else np.minimum(prev_lb, new_lb)
        ub = new_ub if prev_ub is None else np.maximum(prev_ub, new_ub)
        return lb, ub

    def handle_request(self, pkg: ClientPackage):
        action_map: dict[Actions, Callable[[ClientPackage], Any]] = {
            Actions.PUSH_INIT: self._return_save_status,
            Actions.PUSH_UPDATE: self._return_save_status,
            Actions.PULL_INIT: self._return_init_data,
            Actions.PULL_UPDATE: self._return_latest_update,
        }

        action = action_map.get(pkg.action)
        if action:
            return action(pkg)
        raise ValueError(f"Unknown action: {pkg.action}")

    def _return_save_status(self, pkg: ClientPackage):
        if pkg.action == Actions.PUSH_INIT:
            self.dim = pkg.data['dim']
            self.client_bounds.append(pkg.data['bound'])
            self.create_d_share()
        else:
            self.clients_data.update_src(pkg.cid, pkg.version, pkg)
        return 'success'

    def _return_init_data(self, pkg: ClientPackage):
        if self.d_share is None:
            return None
        return {'d_share': self.d_share, 'theta': 5}

    def _return_latest_update(self, pkg: ClientPackage):
        res = self.clients_data.get_res(pkg.cid, pkg.version)
        return res

    def aggregate(self):
        if self.should_agg:
            ver = self.clients_data.available_src_ver
            ls_mat = self._cal_ls_mat_by_rank(ver)
            self.done_vers.add(ver)
            logger.debug(f"ls_mat shape {ls_mat.shape}\n done vers {self.done_vers}")
            for cid, ls_array in zip(self.sorted_ids, ls_mat):
                selected_ids = self._select_updates_for_merge(ls_array)
                res = self._fedavg(selected_ids, ver)
                self.clients_data.update_res(cid, ver, res)
                logger.debug(f"Aggregated client {cid} ver {ver} result, shape={res.shape}")

    @property
    def should_agg(self) -> bool:
        a = self.clients_data.num_clients == self.num_clients
        b = self.clients_data.available_src_ver not in self.done_vers
        return a and b

    def _cal_ls_mat_by_rank(self, version):
        ls_mat = np.zeros((self.num_clients, self.num_clients))
        for cid in self.sorted_ids:
            for _cid in self.sorted_ids:
                if cid == _cid:
                    continue
                update: ClientPackage = self.clients_data.get_src(cid, version)
                _update: ClientPackage = self.clients_data.get_src(_cid, version)
                rank, _rank = update.data['rank'], _update.data['rank']
                for r1, _r1 in zip(rank, _rank):
                    ls_mat[cid-1, _cid-1] += np.sum((r1 < rank) ^ (_r1 < _rank))
        return ls_mat

    def _select_updates_for_merge(self, ls_array):
        clients_number = int(self.num_clients * self.agg_proportion)
        cid_list_of_sorted_array = np.argsort(ls_array) + 1 # cid start from 1
        return cid_list_of_sorted_array[0:clients_number]

    def _fedavg(self, selected_ids, version):
        total_samples = 0
        gp_params = 0
        for cid in selected_ids:
            update_data: ClientPackage = self.clients_data.get_src(cid, version)
            total_samples += update_data.data['size']
            gp_params += update_data.data['size'] * update_data.data['global']
        gp_params /= total_samples
        return gp_params