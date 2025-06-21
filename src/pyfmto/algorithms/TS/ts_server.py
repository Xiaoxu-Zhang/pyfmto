from pyfmto.framework import Server, ClientPackage, ServerPackage
from pyfmto.utilities.tools import update_kwargs


class FtsServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        update_kwargs('FtsServer', self.default_kwargs, kwargs)

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass
