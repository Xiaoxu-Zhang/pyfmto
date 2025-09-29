import numpy as np
import textwrap
from pathlib import Path

from pyfmto.experiments import RunSolutions
from pyfmto.experiments.utils import MergedResults, MetaData
from pyfmto.problems import Solution
from pyfmto.utilities.schemas import STPConfig


class ExpDataGenerator:
    def __init__(self, dim: int, lb: float, ub: float):
        self.dim = dim
        self.lb, self.ub = lb, ub
        self.conf = STPConfig(dim=dim, obj=1, lb=lb, ub=ub)

    def gen_solutions(self, n_solutions: int) -> list[Solution]:
        res = [self.gen_solution() for _ in range(n_solutions)]
        return res

    def gen_solution(self):
        sol = Solution(self.conf)
        x = np.random.uniform(self.lb, self.ub, size=(sol.fe_max, self.dim))
        y = np.random.uniform(1e-5, self.ub, size=(sol.fe_max, 1))
        sol.append(x, y)
        sol._x_global = (self.conf.lb + self.conf.ub) / 2
        sol._y_global = 0.0
        return sol

    def gen_run_data(self, n_tasks: int) -> RunSolutions:
        run_data = RunSolutions()
        for tid in range(n_tasks):
            run_data.update(tid+1, self.gen_solution())
        return run_data

    def gen_runs_data(self, n_tasks: int, n_runs: int) -> list[RunSolutions]:
        return [self.gen_run_data(n_tasks) for _ in range(n_runs)]

    def gen_merged_data(self, n_tasks: int, n_runs: int):
        return MergedResults(self.gen_runs_data(n_tasks, n_runs))

    def gen_metadata(self, algs: list[str], prob: str, npd: str, n_tasks: int, n_runs: int):
        data = {alg: self.gen_merged_data(n_tasks, n_runs) for alg in algs}
        return MetaData(data, prob, npd, Path('tmp'))


def save_module(text, filename: Path):
    filename = filename.with_suffix('.py')
    with open(filename.with_suffix('.py'), 'w') as f:
        f.write(textwrap.dedent(text))


def export_alg_template(name: str):
    """
    Export algorithm template files for client and server components.

    Creates a new algorithm directory with client and server implementation
    templates, including basic structure and placeholder methods.

    Parameters
    ----------
    name : str
        Name of the algorithm to create template for.
    """
    alg_dir = Path(f'algorithms/{name.upper()}')
    alg_dir.mkdir(parents=True, exist_ok=True)
    clt_name = f"{name.title()}Client"
    srv_name = f"{name.title()}Server"
    clt_module = f"{name.lower()}_client"
    srv_module = f"{name.lower()}_server"
    clt_rows = f"""
        import time
        import numpy as np
        from pyfmto.framework import Client, record_runtime, ClientPackage
        from pyfmto.utilities import logger\n\n
        class {clt_name}(Client):
            \"\"\"
            alpha: 0.02
            \"\"\"
            def __init__(self, problem, **kwargs):
                super().__init__(problem)
                kwargs = self.update_kwargs(kwargs)
                self.alpha = kwargs['alpha']
                self.problem.auto_update_solutions = True\n
            def optimize(self):
                x = self.problem.random_uniform_x(1)
                self.problem.evaluate(x)
                time.sleep(self.alpha)\n
    """

    srv_rows = f"""
        from pyfmto.framework import Server, ClientPackage
        from pyfmto.utilities import logger\n\n
        class {srv_name}(Server):
            \"\"\"
            beta: 0.5
            \"\"\"
            def __init__(self, **kwargs):
                super().__init__()
                kwargs = self.update_kwargs(kwargs)
                self.beta = kwargs['beta']
            def handle_request(self, client_data: ClientPackage):
                pass
            def aggregate(self):
                pass
    """

    init_rows = f"""
        from .{srv_module} import {srv_name}
        from .{clt_module} import {clt_name}
    """

    save_module(srv_rows, alg_dir / srv_module)
    save_module(clt_rows, alg_dir / clt_module)
    save_module(init_rows, alg_dir / '__init__')
