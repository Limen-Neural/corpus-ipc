"""
Compact canonical JSON writer with sorted object keys.
"""

from __future__ import annotations

import json

from wire_v1.numbers import format_f64


def dumps(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return _dump_bool(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_f64(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return _dump_list(value)
    if isinstance(value, dict):
        return _dump_object(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _dump_bool(value: bool) -> str:
    if value:
        return "true"
    return "false"


def _dump_list(value: list) -> str:
    return "[" + ",".join(dumps(item) for item in value) + "]"


def _dump_object(value: dict) -> str:
    items = sorted(value.items(), key=_key_bytes)
    body = ",".join(dumps(key) + ":" + dumps(item) for key, item in items)
    return "{" + body + "}"


def _key_bytes(item: tuple[str, object]) -> bytes:
    return item[0].encode("utf-8")
