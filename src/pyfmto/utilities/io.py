from pathlib import Path
from typing import Any, Optional, Union

import msgpack
import numpy as np
from pydantic import validate_call
from ruamel.yaml import YAML, CommentedMap
from ruamel.yaml.error import MarkedYAMLError

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

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
def load_yaml(filename: T_Path, ignore_errors: bool = False):
    try:
        with open(filename) as f:
            return parse_yaml(f.read())
    except Exception:
        if ignore_errors:
            return {}
        raise


def save_yaml(data: Union[dict, CommentedMap], filename: T_Path):
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
    rows = string_stream.getvalue().splitlines()

    add_newline = False
    for line in rows:
        if line.startswith(' '):
            add_newline = True
            break
    if not add_newline:
        return '\n'.join(rows)

    for line in rows:
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
