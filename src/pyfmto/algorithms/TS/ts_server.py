from pyfmto.framework import Server, ClientPackage, ServerPackage


class FtsServer(Server):
    def __init__(self, **kwargs):
        super().__init__()
        self.update_kwargs(kwargs)

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass
