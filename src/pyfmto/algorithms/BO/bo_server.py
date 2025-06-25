from pyfmto.framework import Server, ClientPackage, ServerPackage, Actions
from pyfmto.utilities import logger


class BoServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        self.update_kwargs(kwargs)
        self.versions = {}
        self.ver_ok = -1
        self.set_agg_interval(0.1)

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        if client_data.action == Actions.PUSH_UPDATE:
            return self.handle_push(client_data)
        elif client_data.action == Actions.PULL_UPDATE:
            return self.handle_pull(client_data)
        else:
            raise RuntimeError(f"Unknown action: {client_data.action}")

    def handle_push(self, client_data: ClientPackage) -> ServerPackage:
        self.versions[client_data.cid] = client_data.data
        pkg = ServerPackage('response', 'ok')
        return pkg

    def handle_pull(self, client_data: ClientPackage) -> ServerPackage:
        pkg = ServerPackage('response', self.ver_ok)
        return pkg

    def aggregate(self, client_id):
        vers = list(self.versions.values())
        if len(vers) == self.num_clients:
            self.ver_ok = min(vers)
