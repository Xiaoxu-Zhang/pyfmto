from pathlib import Path
from typing import Any, Optional, Union

import msgpack
import numpy as np
from pydantic import validate_call
from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

T_Path = Union[str, Path]

__all__ = [
    'dumps_yaml',
    'load_msgpack',
    'load_yaml',
    'parse_yaml',
    'recursive_to_pure_dict',
    'save_msgpack',
    'save_yaml',
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


def dumps_yaml(data: dict):
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


def recursive_to_pure_dict(data: dict) -> dict[str, Any]:
    """
    Recursively convert nested dict and CommentedMap objects to a pure Python
    dictionary to avoid YAML serialization issues.

    This function traverses nested dictionaries and converts any CommentedMap
    or CommentedSeq objects to standard Python dict and list objects, making
    the data structure suitable for serialization without ruamel.yaml-specific
    types.

    Args:
        data: A dictionary or CommentedMap object that may contain nested
            CommentedMap, CommentedSeq, or other objects

    Returns:
        A pure Python dictionary with all CommentedMap and CommentedSeq objects
        converted to standard dict and list objects

    Raises:
        TypeError: If an unsupported object type is encountered during conversion
    """

    data = dict(data)
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = recursive_to_pure_dict(dict(v))
        elif isinstance(v, list):
            data[k] = list(map(_to_builtin_type, v))
        elif isinstance(v, str):
            data[k] = str(v)
        elif isinstance(v, bool):
            data[k] = bool(v)
        else:
            data[k] = v
    return data


def _to_builtin_type(v: Any) -> Any:
    for t in [dict, list, tuple, set, int, float, str, bool, bytes, complex]:
        if isinstance(v, t):
            return t(v)
    return v
