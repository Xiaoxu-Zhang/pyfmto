import os
import time
from setproctitle import setproctitle

from pyfmto.problems import Solution
from pyfmto.utilities import (
    logger, reset_log, show_in_table, clear_console,
    backup_log_to)
from .utils import LauncherUtils, RunSolutions
from ..utilities.loaders import ExperimentConfig, ConfigParser

__all__ = ['Launcher']


class Launcher:
    exp_id: int
    exp: ExperimentConfig

    def __init__(self, conf_file: str = 'config.yaml'):
        reset_log()
        clear_console()
        self.conf = ConfigParser(conf_file).launcher

        # Runtime data
        self._repeat_id = 0
        self._running_id = 0
        self._num_clients = 0

    def run(self):
        self._setup()
        for self.exp_id, self.exp in enumerate(self.conf.experiments):
            if self.conf.save:
                self.exp.init_root()
                self.exp.save_info()
            self._repeating()
        self._teardown()

    @staticmethod
    def _setup():
        setproctitle("AlgClients")

    def _repeating(self):
        self._update_repeat_id()
        while not self._finished:
            # Init clients
            problem = self.exp.problem.initialize()
            clt_params = self.exp.algorithm.params.get('client', {})
            clients = [self.exp.algorithm.client(p, **clt_params) for p in problem]
            self._iid_info = problem[0].np_per_dim
            self._num_clients = len(clients)
            self._show_settings()

            # Launch algorithm
            srv_params = self.exp.algorithm.params.get('server', {})
            with LauncherUtils.running_server(self.exp.algorithm.server, **srv_params):
                results = LauncherUtils.start_clients(clients)
            self._save_results(results)
            self._backup_log()
            self._update_repeat_id()
            clear_console()
            time.sleep(1)

    def _backup_log(self):
        reset_log()
        if self.conf.backup:
            backup_log_to(self.exp.root, f'Log of Run {self._repeat_id:02d}.log')

    def _teardown(self):
        clear_console()
        self.conf.show_summary()

    def _show_settings(self):
        n_rep = self.conf.repeat * (self._running_id - 1) + self._repeat_id
        n_all_rep = self.conf.n_exp * self.conf.repeat
        colored_tab, original_tab = show_in_table(
            running=f"{self._running_id}/{self.conf.n_exp}",
            repeat=f"{self._repeat_id}/{self.conf.repeat}",
            progress=f"[{n_rep}/{n_all_rep}][{100 * n_rep / n_all_rep:.2f}%]",
            algorithm=self.exp.algorithm.name,
            problem=self.exp.problem.name,
            iid=self._iid_info,
            clients=self._num_clients,
            save=self.conf.save)
        print(colored_tab)
        logger.info(f"\n{original_tab}")

    def _save_results(self, results: list[tuple[int, Solution]]):
        if self.conf.save:
            file_name = self.exp.root / f"Run {self._repeat_id:02d}.msgpack"
            run_solutions = RunSolutions()
            for cid, solution in results:
                run_solutions.update(cid, solution)
            run_solutions.to_msgpack(file_name)

    def _update_repeat_id(self):
        if self.conf.save:
            self._repeat_id = self._n_results + 1
        else:
            self._repeat_id += 1

    @property
    def _n_results(self) -> int:
        lst_res = [f for f in os.listdir(self.exp.root) if f.endswith(".msgpack")]
        return len(lst_res)

    @property
    def _finished(self):
        return self._repeat_id > self.conf.repeat
