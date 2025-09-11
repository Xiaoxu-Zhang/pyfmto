from pyfmto.framework import Server
from .bo_utils import Actions, ClientPackage


class BoServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        self.update_kwargs(kwargs)
        self.versions = {}
        self.ver_ok = -1
        self.set_agg_interval(0.1)

    def handle_request(self, pkg: ClientPackage):
        if pkg.action == Actions.PUSH_UPDATE:
            return self.handle_push(pkg)
        elif pkg.action == Actions.PULL_UPDATE:
            return self.handle_pull()
        else:
            raise RuntimeError(f"Unknown action: {pkg.action}")

    def handle_push(self, pkg: ClientPackage):
        self.versions[pkg.cid] = pkg.data
        return 'save success'

    def handle_pull(self):
        return self.ver_ok

    def aggregate(self):
        vers = list(self.versions.values())
        if len(vers) == self.num_clients:
            self.ver_ok = min(vers)
