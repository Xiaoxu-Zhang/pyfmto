import msgpack
import numpy as np
from pathlib import Path
from typing import Union, Any, Optional

from pydantic import validate_call
from ruamel.yaml import YAML, CommentedMap
from ruamel.yaml.error import MarkedYAMLError

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.default_flow_style = None

T_Path = Union[str, Path]

__all__ = [
    'load_yaml',
    'save_yaml',
    'parse_yaml',
    'dumps_yaml',
    'load_msgpack',
    'save_msgpack',
]


@validate_call
def load_yaml(filename: T_Path):
    with open(filename, 'r') as f:
        return parse_yaml(f.read())


@validate_call
def save_yaml(data: dict, filename: T_Path):
    with open(filename, 'w') as f:
        f.write(dumps_yaml(data))


@validate_call
def parse_yaml(text: Optional[str]):
    if not text:
        return {}
    lines = [line for line in text.splitlines() if line.strip() != '']
    try:
        return yaml.load('\n'.join(lines))
    except MarkedYAMLError:
        raise


def dumps_yaml(data: Union[dict, CommentedMap]):
    from io import StringIO
    string_stream = StringIO()
    yaml.dump(data, string_stream)
    text: list[str] = []
    for line in string_stream.getvalue().splitlines():
        if text and not line.startswith(' '):
            # Only add a newline if the line is
            # not indented and not the first line
            text.append(f"\n{line}")
        else:
            text.append(line)
    return '\n'.join(text)


@validate_call
def save_msgpack(data: dict, filename: T_Path) -> None:
    with open(filename, 'wb') as f:
        packed = msgpack.packb(data, default=_encode_hook, use_bin_type=True)
        f.write(packed)


@validate_call
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
