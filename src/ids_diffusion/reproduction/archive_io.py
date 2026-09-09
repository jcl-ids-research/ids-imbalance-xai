"""Narrow archived JSON to typed values at the point it is read.

Archived results are written by the server scripts and arrive as untyped JSON.
Parsing them here, once, means the rest of the reproduction package works with
numbers and mappings rather than with `Any`, and a malformed archive is reported
against the file that contains it instead of failing somewhere downstream.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ids_diffusion.errors import DatasetFileError

JsonValue = float | int | str | bool | None | Mapping[str, "JsonValue"] | list["JsonValue"]
JsonObject = Mapping[str, JsonValue]


def read_json(path: Path) -> JsonObject:
    """Load one archived result, failing loudly when it is absent."""
    if not path.is_file():
        raise DatasetFileError(path=str(path), detail="archived result not found")
    parsed: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    return as_object(parsed, path, "archived result")


def as_object(value: JsonValue, path: Path, what: str) -> JsonObject:
    """Narrow one JSON value to an object, naming the file when it is not."""
    if not isinstance(value, Mapping):
        raise DatasetFileError(path=str(path), detail=f"{what} is not an object")
    return value


def as_float(value: JsonValue, path: Path, what: str) -> float:
    """Narrow one JSON value to a number, naming the file when it is not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetFileError(path=str(path), detail=f"{what} is not a number")
    return float(value)


def as_list(value: JsonValue, path: Path, what: str) -> list[JsonValue]:
    """Narrow one JSON value to a list, naming the file when it is not."""
    if not isinstance(value, list):
        raise DatasetFileError(path=str(path), detail=f"{what} is not a list")
    return value


def nested(block: JsonObject, *keys: str) -> JsonValue:
    """Walk a chain of keys, returning None as soon as one is absent."""
    current: JsonValue = block
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def nested_float(block: JsonObject, path: Path, *keys: str) -> float | None:
    """Return a nested number, or None when the chain does not reach one."""
    value = nested(block, *keys)
    return None if value is None else as_float(value, path, ".".join(keys))


__all__ = [
    "JsonObject",
    "JsonValue",
    "as_float",
    "as_list",
    "as_object",
    "nested",
    "nested_float",
    "read_json",
]
