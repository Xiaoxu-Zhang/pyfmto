import atexit
import os
import time
from pathlib import Path
from setproctitle import setproctitle

from pyfmto.algorithms import load_algorithm, get_alg_kwargs
from pyfmto.problems import load_problem, Solution
from pyfmto.utilities.schemas import LauncherConfig
from pyfmto.utilities import (
    logger, reset_log, show_in_table, clear_console,
    backup_log_to, load_yaml, save_yaml)
from .utils import LauncherUtils, RunSolutions

__all__ = ['Launcher']


class Launcher:
    def __init__(self, conf_file: str = 'config.yaml'):
        reset_log()
        clear_console()
        LauncherUtils.kill_server()
        all_conf = load_yaml(conf_file)
        launcher_conf = LauncherConfig(**all_conf.get('launcher'))
        self.combinations = LauncherUtils.gen_exp_combinations(
            launcher_conf=launcher_conf,
            alg_conf=all_conf.get('algorithms', {}),
            prob_conf=all_conf.get('problems', {}))
        # Launcher settings
        self.repeat = launcher_conf.repeat
        self.save = launcher_conf.save
        self.seed = launcher_conf.seed
        self.backup = launcher_conf.backup

        # Runtime data
        self._iid_info = 0
        self._repeat_id = 0
        self._running_id = 0
        self._num_clients = 0
        self._alg = ''
        self._alg_alias = ''
        self._prob = ''
        self._res_dir = Path(launcher_conf.results)
        self._prob_args = {}
        self._clt_kwargs = {}
        self._srv_kwargs = {}

        atexit.register(LauncherUtils.kill_server)

    def run(self):
        self._setup()
        for idx, args in enumerate(self.combinations):
            self._initializing(idx, *args)
            self._save_kwargs()
            self._repeating()
        self._teardown()

    @staticmethod
    def _setup():
        setproctitle("AlgClients")

    def _initializing(self, idx, alg_alias, alg_args, prob_name, prob_args):
        self._running_id = idx + 1
        self._alg = alg_args.get('base', alg_alias)
        self._alg_alias = alg_alias
        self._clt_kwargs = alg_args.get('client', {})
        self._srv_kwargs = alg_args.get('server', {})
        self._prob = prob_name
        self._prob_args = prob_args
        self._res_dir = LauncherUtils.gen_path(alg_alias, prob_name, prob_args)
        self._repeat_id = 0

    def _save_kwargs(self):
        if not self.save:
            return
        kwargs = {}
        default_kwargs = get_alg_kwargs(self._alg)
        if self._clt_kwargs:
            kwargs.update(client=self._clt_kwargs)
        else:
            kwargs.update(client=default_kwargs.get('client', {}))
        if self._srv_kwargs:
            kwargs.update(server=self._srv_kwargs)
        else:
            kwargs.update(server=default_kwargs.get('server', {}))
        fdir = self._res_dir.parents[1]
        fdir.mkdir(parents=True, exist_ok=True)
        save_yaml(kwargs, fdir / "arguments.yaml")

    def _repeating(self):
        alg_modules = load_algorithm(self._alg)
        client_cls = alg_modules['client']
        server_cls = alg_modules['server']
        self._update_repeat_id()
        while not self._finished:
            # Init clients
            problem = load_problem(self._prob, **self._prob_args)
            clients = [client_cls(p, **self._clt_kwargs) for p in problem]
            self._iid_info = problem[0].np_per_dim
            self._num_clients = len(clients)
            self._show_settings()

            # Launch algorithm
            LauncherUtils.start_server(server_cls, **self._srv_kwargs)
            results = LauncherUtils.start_clients(clients)
            self._save_results(results)
            self._update_repeat_id()
            reset_log()
            self._backup_log()
            clear_console()
            time.sleep(1)

    def _backup_log(self):
        dest_dir = self._res_dir.parent.parent
        if self.backup:
            backup_log_to(dest_dir)

    def _teardown(self):
        clear_console()
        print('All runs finished.')
        colored_tab, _ = show_in_table(
            exp_total=self._num_comb,
            rep_per_exp=self.repeat,
            rep_total=self._num_comb * self.repeat,
        )
        print(colored_tab)

    def _show_settings(self):
        n_rep = self.repeat * (self._running_id - 1) + self._repeat_id
        n_all_rep = self._num_comb * self.repeat
        colored_tab, original_tab = show_in_table(
            running=f"{self._running_id}/{self._num_comb}",
            repeat=f"{self._repeat_id}/{self.repeat}",
            progress=f"[{n_rep}/{n_all_rep}][{100 * n_rep / n_all_rep:.2f}%]",
            algorithm=self._alg_alias,
            problem=self._prob,
            iid=self._iid_info,
            clients=self._num_clients,
            save=self.save)
        print(colored_tab)
        logger.info(f"\n{original_tab}")

    def _save_results(self, results: list[tuple[int, Solution]]):
        if self.save:
            res_path = Path(self._res_dir)
            file_name = res_path / f"Run {self._repeat_id}.msgpack"
            run_solutions = RunSolutions()
            for cid, solution in results:
                run_solutions.update(cid, solution)
            run_solutions.to_msgpack(file_name)

    def _update_repeat_id(self):
        if self.save:
            self._repeat_id = self._n_results + 1
        else:
            self._repeat_id += 1

    @property
    def _n_results(self) -> int:
        res_root = Path(self._res_dir)
        res_root.mkdir(parents=True, exist_ok=True)
        return len(os.listdir(res_root))

    @property
    def _num_comb(self):
        return len(self.combinations)

    @property
    def _finished(self):
        return self._repeat_id > self.repeat
