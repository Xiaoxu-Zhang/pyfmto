import textwrap
from pathlib import Path


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
