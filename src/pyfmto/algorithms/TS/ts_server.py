from pyfmto.framework import Server, ClientPackage, ServerPackage


class FtsServer(Server):
    def __init__(self):
        super().__init__()

    def handle_request(self, client_data: ClientPackage) -> ServerPackage:
        pass

    def aggregate(self, client_id):
        pass
