from pathlib import Path

from .client import Client, record_runtime
from .server import Server
from .packages import *
from ..utilities import colored


def _to_module(rows, filename: Path):
    rows = [f'{row}\n' for row in rows]
    filename = filename.with_suffix('.py')
    if filename.exists():
        print(f"{colored('Skipped', 'yellow')} existing file {filename}")
        return
    with open(filename.with_suffix('.py'), 'w') as f:
        f.writelines(rows)
    print(f"{colored('Created', 'green')} {filename}")

def export_alg_template(name: str):
    alg_dir = Path(f'algorithms/{name.upper()}')
    alg_dir.mkdir(parents=True, exist_ok=True)
    srv_name = f"{name.title()}Server"
    clt_name = f"{name.title()}Client"
    srv_module = f"{name.lower()}_server"
    clt_module = f"{name.lower()}_client"
    srv_rows = [
        "from pyfmto.framework import Server, ClientPackage, ServerPackage, Actions",
        "from pyfmto.utilities import logger\n\n",
        f"class {srv_name}(Server):",
        "    \"\"\"",
        "    alpha: 0.5",
        "    \"\"\"",
        "    def __init__(self, **kwargs):",
        "        super().__init__()",
        "        kwargs = self.update_kwargs(kwargs)",
        "        self.alpha = kwargs['alpha']\n",
        "    def handle_request(self, client_data: ClientPackage) -> ServerPackage:",
        "        pass\n",
        "    def aggregate(self, client_id):",
        "        pass"
    ]

    clt_rows = [
        "from pyfmto.framework import Client, record_runtime, ClientPackage, Actions, ServerPackage",
        "from pyfmto.utilities import logger\n\n",
        f"class {clt_name}(Client):",
        "    \"\"\"",
        "    gamma: 0.5",
        "    \"\"\"",
        "    def __init__(self, problem, **kwargs):",
        "        super().__init__(problem)",
        "        kwargs = self.update_kwargs(kwargs)",
        "        self.gamma = kwargs['gamma']\n",
        "    def optimize(self):",
        "        pass"
    ]

    init_rows = [
        f"from .{srv_module} import {srv_name}",
        f"from .{clt_module} import {clt_name}"
    ]

    _to_module(srv_rows, alg_dir / srv_module)
    _to_module(clt_rows, alg_dir / clt_module)
    _to_module(init_rows, alg_dir / '__init__')
