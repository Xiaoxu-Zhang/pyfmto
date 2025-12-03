import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from setproctitle import setproctitle

from pyfmto.utilities import (
    logger,
    show_in_table,
    clear_console,
    titled_tabulate,
    terminate_popen,
    tabulate_formats as tf,
)
from .utils import RunSolutions
from ..framework import Client
from ..utilities.loaders import ExperimentConfig, ConfigLoader

__all__ = ['Launcher']


class Launcher:
    exp_idx: int
    exp: ExperimentConfig
    clients: list[Client]

    def __init__(self, conf_file: str = 'config.yaml'):
        clear_console()
        self.conf = ConfigLoader(conf_file).launcher

        # Runtime data
        self._repeat_id = 0

    def run(self):
        self._setup()
        for self.exp_idx, self.exp in enumerate(self.conf.experiments):
            if self.conf.save:
                self.exp.init_root()
                self.exp.backup_params()
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
            self.clients = [self.exp.algorithm.client(p, **clt_params) for p in problem]
            self._iid_info = problem[0].npd
            self._show_progress()

            # Launch algorithm
            with self.running_server():
                results = self.start_clients()
            self._save_results(results)
            self._save_rounds_info(results)
            self._update_repeat_id()
            clear_console()
            time.sleep(1)

    @contextmanager
    def running_server(self):
        server = self.exp.algorithm.server
        kwargs = self.exp.algorithm.params.get('server', {})
        module_name = server.__module__
        class_name = server.__name__

        cmd = [
            sys.executable, "-c",
            f"from {module_name} import {class_name}; "
            f"srv = {class_name}(**{repr(kwargs)}); "
            f"srv.start()"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Server started.")
        time.sleep(3)
        try:
            yield process
        finally:
            terminate_popen(process)
            logger.debug("Server terminated.")

    def start_clients(self) -> list[Client]:
        pool = ThreadPoolExecutor(max_workers=len(self.clients))
        futures = [pool.submit(c.start) for c in self.clients]
        pool.shutdown(wait=True)
        return [fut.result() for fut in futures]

    def _save_rounds_info(self, results: list[Client]):
        if self.conf.backup:
            tables = {clt.id: clt for clt in results}
            info_str = ''
            for cid in sorted(tables.keys()):
                clt = tables[cid]
                tab = titled_tabulate(
                    f"{clt.name} {clt.problem.name}({clt.dim}D)",
                    '=', clt.rounds_info, headers='keys', tablefmt=tf.rounded_grid
                )
                info_str = f"{info_str}{tab}\n"
            res_file = self.exp.result_name(self._repeat_id)
            log_dir = res_file.with_name('rounds_info')
            log_name = res_file.with_suffix('.log').name
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / log_name, 'w') as f:
                f.write(info_str)

    def _teardown(self):
        clear_console()
        self.conf.show_summary()

    def _show_progress(self):
        curr_rep = self.conf.repeat * self.exp_idx + self._repeat_id - 1
        total_rep = self.conf.total_repeat
        colored_tab, original_tab = show_in_table(
            running=f"{self.exp_idx+1}/{self.conf.n_exp}",
            repeat=f"{self._repeat_id}/{self.conf.repeat}",
            progress=f"[{curr_rep}/{total_rep}][{100 * curr_rep / total_rep:.2f}%]",
            algorithm=self.exp.algorithm.name,
            problem=self.exp.problem.name,
            iid=self._iid_info,
            clients=len(self.clients),
            save=self.conf.save)
        print(colored_tab)
        logger.info(f"\n{original_tab}")

    def _save_results(self, results: list[Client]):
        if self.conf.save:
            run_solutions = RunSolutions()
            for clt in results:
                run_solutions.update(clt.id, clt.solutions)
            run_solutions.to_msgpack(self.exp.result_name(self._repeat_id))

    def _update_repeat_id(self):
        if self.conf.save:
            self._repeat_id = self.exp.num_results + 1
        else:
            self._repeat_id += 1

    @property
    def _finished(self):
        return self._repeat_id > self.conf.repeat
