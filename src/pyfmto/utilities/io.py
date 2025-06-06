import msgpack
import numpy as np
import yaml
from pathlib import Path
from typing import Union, Any
from yaml import MarkedYAMLError

T_Path = Union[str, Path]

__all__ = [
    'load_yaml',
    'save_msgpack',
    'load_msgpack'
]


def load_yaml(path: T_Path):
    path = Path(path)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except MarkedYAMLError:
            raise
    else:
        raise FileNotFoundError(f'File {path} does not exist.')


def save_msgpack(data: dict, filename: T_Path) -> None:
    with open(filename, 'wb') as f:
        packed = msgpack.packb(data, default=_encode_hook, use_bin_type=True)
        f.write(packed)


def load_msgpack(filename: T_Path) -> dict:
    with open(filename, 'rb') as f:
        data = msgpack.unpackb(f.read(), object_hook=_decode_hook, raw=False, strict_map_key=False)
    return data


def _encode_hook(obj: Any) -> dict:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": obj.dtype.name,
            "shape": obj.shape,
            "data": obj.tobytes()
        }
    elif isinstance(obj, set):
        return {
            "__set__": True,
            "items": list(obj)
        }
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")


def _decode_hook(obj: dict) -> Any:
    if "__ndarray__" in obj:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        data = obj["data"]
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    elif "__set__" in obj:
        return set(obj["items"])
    return obj