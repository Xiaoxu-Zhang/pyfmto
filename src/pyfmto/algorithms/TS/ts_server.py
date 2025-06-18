from pyfmto.framework import Server, ClientPackage, ServerPackage
from pyfmto.utilities.tools import warn_unused_kwargs


class FtsServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        warn_unused_kwargs('FtsServer', kwargs)

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass
