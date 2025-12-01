import time
from setproctitle import setproctitle

from pyfmto.utilities import logger, show_in_table, clear_console, titled_tabulate, tabulate_formats as tf
from .utils import LauncherUtils, RunSolutions
from ..framework import Client
from ..utilities.loaders import ExperimentConfig, ConfigLoader

__all__ = ['Launcher']


class Launcher:
    exp_idx: int
    exp: ExperimentConfig

    def __init__(self, conf_file: str = 'config.yaml'):
        clear_console()
        self.conf = ConfigLoader(conf_file).launcher

        # Runtime data
        self._repeat_id = 0
        self._num_clients = 0

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
            clients = [self.exp.algorithm.client(p, **clt_params) for p in problem]
            self._iid_info = problem[0].npd
            self._num_clients = len(clients)
            self._show_progress()

            # Launch algorithm
            srv_params = self.exp.algorithm.params.get('server', {})
            with LauncherUtils.running_server(self.exp.algorithm.server, **srv_params):
                results = LauncherUtils.start_clients(clients)
            self._save_results(results)
            self._save_rounds_info(results)
            self._update_repeat_id()
            clear_console()
            time.sleep(1)

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
            clients=self._num_clients,
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
