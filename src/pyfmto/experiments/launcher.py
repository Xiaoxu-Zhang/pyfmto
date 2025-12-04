import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from rich import box, progress as rpg
from rich.console import Group, Console
from rich.live import Live
from rich.table import Table
from setproctitle import setproctitle
from tabulate import tabulate

from pyfmto.utilities import (
    logger,
    titled_tabulate,
    terminate_popen,
    tabulate_formats as tf, clear_console,
)
from .utils import RunSolutions
from ..framework import Client
from ..utilities.loaders import ExperimentConfig, ConfigLoader

__all__ = ['Launcher']


class Launcher:
    exp_idx: int
    exp: ExperimentConfig
    clients: list[Client]
    table: Table
    progress: rpg.Progress

    def __init__(self, conf_file: str = 'config.yaml'):
        self.conf = ConfigLoader(conf_file).launcher

        # Runtime data
        self._repeat_id = 0
        self._results: list[Client] = []

    def run(self):
        self._setup()
        for self.exp_idx, self.exp in enumerate(self.conf.experiments):
            logger.info(f"\n{self.exp}")
            if self.conf.save:
                self.exp.init_root()
                self.exp.backup_params()
            self._repeating()
        self.conf.show_summary()

    def _setup(self):
        self.progress = rpg.Progress(
            rpg.TextColumn("[progress.description]{task.description}"),
            rpg.BarColumn(),
            rpg.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            rpg.TimeRemainingColumn(),
        )
        setproctitle("AlgClients")

    def _repeating(self):
        self._repeat_id = 0
        self._update_repeat_id()
        while not self._finished:
            self._init_clients()
            self._launch_exp()
            self._save_results()
            self._save_rounds_info()
            self._update_repeat_id()
            time.sleep(1)
        self.exp.success = True

    def _init_clients(self):
        problem = self.exp.problem.initialize()
        clt_params = self.exp.algorithm.params.get('client', {})
        self.clients = [self.exp.algorithm.client(p, **clt_params) for p in problem]

    def _launch_exp(self):
        clear_console()
        with Live(self._panel, console=Console(width=120), refresh_per_second=10):
            with self._running_server():
                self._start_clients()

    @property
    def _panel(self) -> Group:
        self._remove_bars()
        self._update_table()
        return Group(self.table, self.progress)

    def _remove_bars(self):
        for bar in self.progress.task_ids:
            self.progress.remove_task(bar)

    def _update_table(self):
        curr_rep = self.conf.repeat * self.exp_idx + self._repeat_id
        total_rep = self.conf.total_repeat
        tab_dict = {
            'running': f"{self.exp_idx+1}/{self.conf.n_exp}",
            'repeating': f"{self._repeat_id}/{self.conf.repeat}",
            'progress': f"[{curr_rep}/{total_rep}][{100 * curr_rep / total_rep:.2f}%]",
            'algorithm': self.exp.algorithm.name,
            'problem': self.exp.problem.name,
            'NPD': self.clients[0].problem.npd,
            'clients': len(self.clients),
            'save': self.conf.save
        }

        def mapper(v):
            if v is True:
                return '[green]yes[/green]'
            elif v is False:
                return '[red]no[/red]'
            elif isinstance(v, int):
                return f'[magenta]{v}[/magenta]'
            else:
                return str(v)

        keys = list(tab_dict.keys())
        values = list(tab_dict.values())
        colored_values = list(map(mapper, values))

        tab_csl = Table(box=box.ROUNDED)
        [tab_csl.add_column(k, justify='center') for k in keys]
        tab_csl.add_row(*colored_values)
        self.table = tab_csl

        alignment = ['center'] * len(keys)
        original_tab = {k: [v] for k, v in zip(keys, values)}
        tab_log = tabulate(original_tab, headers='keys', tablefmt='rounded_grid', colalign=alignment)
        logger.info(f"\n{tab_log}")

    @contextmanager
    def _running_server(self):
        server = self.exp.algorithm.server
        kwargs = self.exp.algorithm.params.get('server', {})

        cmd = [
            sys.executable, "-c",
            f"from {server.__module__} import {server.__name__}; "
            f"srv = {server.__name__}(**{repr(kwargs)}); "
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

    def _start_clients(self):
        n_clients = len(self.clients)
        for c in self.clients:
            bar = self.progress.add_task(c.name, total=c.fe_max)
            c.progress = self.progress
            c.bar = bar
        pool = ThreadPoolExecutor(max_workers=n_clients)
        futures = [pool.submit(c.start) for c in self.clients]
        pool.shutdown(wait=True)
        self._results = [fut.result() for fut in futures]

    def _save_rounds_info(self):
        if self.conf.backup:
            tables = {clt.id: clt for clt in self._results}
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

    def _save_results(self):
        if self.conf.save:
            run_solutions = RunSolutions()
            for clt in self._results:
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
