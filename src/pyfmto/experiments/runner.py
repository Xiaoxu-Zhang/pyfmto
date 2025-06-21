import atexit
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pyfmto.algorithms import load_algorithm
from pyfmto.problems import load_problem
from pyfmto.utilities import logger, reset_log, timer, show_in_table
from setproctitle import setproctitle
from subprocess import Popen
from typing import Optional

from .utils import clear_console, prepare_server, gen_path, check_path, save_results, load_experiments

__all__ = ['exp']


class Runner:
    def __init__(self):
        comb, settings = load_experiments()
        self.combinations = comb
        self.serv_proc: Optional[Popen] = None

        # other settings
        self.num_runs = settings.get('num_runs', 3)
        self.res_dir = settings.get('results')
        self.save_res = settings.get('save_res', True)
        self.clean_tmp = settings.get('clean_tmp', True)
        self.seed = settings.get('seed', 42)
        atexit.register(self.stop_server)

    def run(self):
        setproctitle("AlgClients")
        for idx, (alg_name, alg_args, prob_name, prob_args) in enumerate(self.combinations):
            res_path = gen_path(alg_name, prob_name, prob_args)
            self.iterating(idx+1, alg_name, alg_args, prob_name, prob_args, res_path)
        clear_console()

        print(f'All runs finished.')
        colored_tab, _ = show_in_table(
            comb=len(self.combinations),
            runs_per_comb=self.num_runs,
            runs_total=len(self.combinations) * self.num_runs,
        )
        if self.clean_tmp:
            os.remove("temp_server.py")
        print(colored_tab)

    def iterating(self, comb_id, alg_name, alg_args, prob_name, prob_args, res_path):
        clt_kwargs = alg_args.get('client', {})
        srv_kwargs = alg_args.get('server', {})
        prepare_server(alg_name, **srv_kwargs)
        client_cls = load_algorithm(alg_name).get('client')
        curr_run = self.update_iter(None, res_path)
        while curr_run <= self.num_runs:
            reset_log()
            clear_console()

            # Init clients
            problem = load_problem(prob_name, **prob_args)
            clients = [client_cls(p, **clt_kwargs) for p in problem]

            # Show settings
            colored_tab, original_tab = show_in_table(
                comb=f"{comb_id}/{self.num_comb}", runs=f"{curr_run}/{self.num_runs}",
                algorithm=alg_name, problem=problem.name, iid=problem[0].np_per_dim,
                clients=problem.task_num,save=self.save_res)
            print(colored_tab)
            logger.info(f"\n{original_tab}")

            # Run
            self.start_server()
            c_res = self.start_clients(clients)
            if self.save_res:
                save_results(c_res, res_path, curr_run)
            curr_run = self.update_iter(curr_run, res_path)
            time.sleep(1)

    def start_server(self) -> int:
        """
        Start the server process.
        Returns
        -------
        pid : int
            Server process id.
        """
        if os.name == 'posix':
            proc = subprocess.Popen(args=["python", "temp_server.py"], start_new_session=True,
                                    stdin=subprocess.DEVNULL)
        elif os.name == 'nt':
            proc = subprocess.Popen(args=["python", "temp_server.py"], creationflags=subprocess.CREATE_NEW_CONSOLE,
                                    stdin=subprocess.DEVNULL)
        else:
            raise OSError(f"Unsupported operating system: {os.name}")
        logger.info("Server started.")
        time.sleep(2)
        self.serv_proc = proc
        return os.getpgid(proc.pid)

    @staticmethod
    @timer("Whole run")
    def start_clients(clients):
        thread_pool = ThreadPoolExecutor(max_workers=len(clients))
        client_futures = [thread_pool.submit(c.start) for c in clients]
        thread_pool.shutdown(wait=True)
        return (c.result() for c in client_futures)

    def update_iter(self, curr_run, res_path):
        if self.save_res:
            num_res = check_path(res_path)
            return num_res + 1
        elif curr_run is None:
            return 1
        else:
            return curr_run + 1

    def stop_server(self):
        if self.serv_proc is None:
            return
        try:
            if os.name == 'posix':
                pid = os.getpgid(self.serv_proc.pid)
                os.killpg(pid, 15)
            else:
                self.serv_proc.terminate()
        except Exception as e:
            logger.warning(f"Failed to terminate process {self.serv_proc.pid}: {e}")

    @property
    def num_comb(self):
        return len(self.combinations)

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

exp = Runner()