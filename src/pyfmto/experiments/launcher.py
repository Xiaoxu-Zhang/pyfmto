import atexit
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ruamel.yaml import YAML
from setproctitle import setproctitle
from subprocess import Popen
from typing import Optional
from yaml import safe_load

from pyfmto.algorithms import load_algorithm, get_alg_kwargs
from pyfmto.problems import load_problem
from pyfmto.utilities import logger, reset_log, timer, show_in_table, backup_log_to
from .utils import (
    clear_console,
    gen_path,
    check_path,
    kill_server,
    save_results,
    gen_exp_combinations,
    load_launcher_settings)

__all__ = ['Launcher']


class Launcher:
    """
    repeat: 3         # number of runs repeating
    backup: True      # backup log file to results directory
    dir: out/results  # dir of results
    save: True        # save results
    seed: 42          # random seed
    """
    def __init__(self):
        reset_log()
        clear_console()
        kill_server()
        settings = load_launcher_settings()
        self.combinations = gen_exp_combinations(settings)
        self.serv_proc: Optional[Popen] = None

        # Launcher settings
        default_settings = safe_load(self.__class__.__doc__)
        for k in default_settings.keys():
            v = settings.get(k)
            if v is not None:
                default_settings[k] = v
        self.repeat = default_settings['repeat']
        self.dir    = default_settings['dir']
        self.save   = default_settings['save']
        self.seed   = default_settings['seed']
        self.backup = default_settings['backup']

        # Runtime data
        self._iid_info = 0
        self._repeat_id = 0
        self._running_id = 0
        self._num_clients = 0
        self._alg = ''
        self._alg_alias = ''
        self._prob = ''
        self._results = []
        self._res_dir = Path.cwd()
        self._alg_args = {}
        self._prob_args = {}
        self._clt_kwargs = {}
        self._srv_kwargs = {}

        atexit.register(kill_server)

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
        self._alg_args = alg_args
        self._clt_kwargs = alg_args.get('client', {})
        self._srv_kwargs = alg_args.get('server', {})
        self._prob = prob_name
        self._prob_args = prob_args
        self._res_dir = gen_path(alg_alias, prob_name, prob_args)
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
        yaml = YAML()
        filename = fdir / f"arguments.yaml"
        if not filename.exists():
            with open(filename, 'w') as f_bac:
                yaml.dump(kwargs, f_bac)

    def _repeating(self):
        client_cls = load_algorithm(self._alg)['client']
        self._update_repeat_id()
        while not self._finished:

            # Init clients
            problem = load_problem(self._prob, **self._prob_args)
            clients = [client_cls(p, **self._clt_kwargs) for p in problem]
            self._iid_info = problem[0].np_per_dim
            self._num_clients = len(clients)
            self._show_settings()

            # Launch algorithm
            self._start_server()
            self._start_clients(clients)
            self._save_results()
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
        print(f'All runs finished.')
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
            progress=f"[{n_rep}/{n_all_rep}][{100*n_rep/n_all_rep:.2f}%]",
            algorithm=self._alg_alias,
            problem=self._prob,
            iid=self._iid_info,
            clients=self._num_clients,
            save=self.save)
        print(colored_tab)
        logger.info(f"\n{original_tab}")

    def _start_server(self):
        """
        Start the server process.
        """
        srv_cls = load_algorithm(self._alg)['server']
        module_name = srv_cls.__module__
        class_name = srv_cls.__name__

        cmd = [
            "python", "-c",
            f"from {module_name} import {class_name}; "
            f"srv = {class_name}(**{repr(self._srv_kwargs)}); "
            f"srv.start()"
        ]

        if os.name == 'posix':
            subprocess.Popen(cmd,
                             start_new_session=True,
                             stdin=subprocess.DEVNULL)
        elif os.name == 'nt':
            subprocess.Popen(cmd,
                             creationflags=subprocess.CREATE_NEW_CONSOLE,
                             stdin=subprocess.DEVNULL)
        else:
            raise OSError(f"Unsupported operating system: {os.name}")
        logger.info("Server started.")
        time.sleep(2)

    @timer("Whole run")
    def _start_clients(self, clients):
        thread_pool = ThreadPoolExecutor(max_workers=len(clients))
        client_futures = [thread_pool.submit(c.start) for c in clients]
        thread_pool.shutdown(wait=True)
        self._results = (c.result() for c in client_futures)

    def _save_results(self):
        if self.save:
            save_results(
                self._results,
                self._res_dir,
                self._repeat_id
            )

    def _update_repeat_id(self):
        if self.save:
            self._repeat_id = check_path(self._res_dir) + 1
        else:
            self._repeat_id += 1

    @property
    def _num_comb(self):
        return len(self.combinations)

    @property
    def _finished(self):
        return self._repeat_id > self.repeat


SETTING_YML = """
results: out/results

runs:
  num_runs: 3
  save_res: True
  clean_tmp: True
  algorithms: [FDEMD, FMTBO]
  problems: [tetci2019, tevc2024]

analyses:
  algorithms:
    - [FMTBO, FDEMD, ADDFBO]
  problems: [tetci2019, cec2022]

algorithms:
  FDEMD:
    client:
      max_gen: 20
    server:
      ensemble_size: 20

problems:
  tevc2024:
    args:
      src_problem: [Ackley, Ellipsoid]
      np_per_dim: [1, 2, 4, 6]

"""


if not os.path.exists('settings.yaml'):
    with open('settings.yaml', 'w') as f:
        f.write(SETTING_YML)
