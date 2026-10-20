"""Compact canonical JSON writer with sorted object keys."""

from __future__ import annotations

import json

from wire_v1.numbers import format_f64


def dumps(value: object) -> str:
    return _select(value)(value)


def _select(value: object):
    if value is None:
        return _dump_null
    return _select_scalar(value)


def _select_scalar(value: object):
    if isinstance(value, bool):
        return _dump_bool
    if isinstance(value, int):
        return str
    return _select_text(value)


def _select_text(value: object):
    if isinstance(value, float):
        return format_f64
    if isinstance(value, str):
        return _dump_str
    return _select_container(value)


def _select_container(value: object):
    if isinstance(value, list):
        return _dump_list
    if isinstance(value, dict):
        return _dump_object
    raise TypeError(f"cannot encode {type(value).__name__}")


def _dump_null(_value: object) -> str:
    return "null"


def _dump_bool(value: bool) -> str:
    if value:
        return "true"
    return "false"


def _dump_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dump_list(value: list) -> str:
    return "[" + ",".join(dumps(item) for item in value) + "]"


def _dump_object(value: dict) -> str:
    items = sorted(value.items(), key=_key_bytes)
    body = ",".join(dumps(key) + ":" + dumps(item) for key, item in items)
    return "{" + body + "}"


def _key_bytes(item: tuple[str, object]) -> bytes:
    return item[0].encode("utf-8")
