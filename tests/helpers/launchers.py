import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from pyfmto.framework import Client
from pyfmto.utilities.tools import terminate_popen


@contextmanager
def running_server(server, **kwargs):
    module_name = server.__module__
    class_name = server.__name__

    cmd = [
        sys.executable, "-c",
        f"from {module_name} import {class_name}; "
        f"srv = {class_name}(**{kwargs!r}); "
        f"srv.start()"
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)
    try:
        yield process
    finally:
        terminate_popen(process)


def start_clients(clients: list[Client]) -> list[Client]:
    pool = ThreadPoolExecutor(max_workers=len(clients))
    futures = [pool.submit(c.start) for c in clients]
    pool.shutdown(wait=True)
    return [fut.result() for fut in futures]
