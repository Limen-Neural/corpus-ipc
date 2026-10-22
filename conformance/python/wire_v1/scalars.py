"""Integer, string, and collection checks shared by the decoders."""

from __future__ import annotations

import math

from wire_v1.errors import payload_error
from wire_v1.limits import MAX_AGGREGATE, MAX_STRING_BYTES
from wire_v1.numbers import format_f64


def string_bytes(value: str, path: str) -> None:
    if len(value.encode("utf-8")) > MAX_STRING_BYTES:
        raise payload_error(f"{path} exceeds {MAX_STRING_BYTES} bytes")


def check_count(actual: int, limit: int, path: str) -> None:
    if actual > limit:
        raise payload_error(f"{path} length {actual} exceeds {limit}")


def check_finite(value: float, path: str) -> None:
    if not math.isfinite(value):
        raise payload_error(f"{path} is non-finite")


def check_range(value: float, low: float, high: float, path: str) -> None:
    check_finite(value, path)
    if value < low:
        _out_of_range(value, low, high, path)
    if value > high:
        _out_of_range(value, low, high, path)


def _out_of_range(value: float, low: float, high: float, path: str) -> None:
    shown = format_f64(value)
    raise payload_error(f"{path} {shown} is outside [{low}, {high}]")


def add_total(total: int, count: int, path: str) -> int:
    total += count
    if total > MAX_AGGREGATE:
        raise payload_error(f"{path} aggregate exceeds {MAX_AGGREGATE}")
    return total


def require_object(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        raise payload_error(f"{path} must be a JSON object")
    return value


def require_key(obj: dict, key: str, path: str) -> object:
    if key not in obj:
        raise payload_error(f"missing {path}")
    return obj[key]


def as_u64(value: object, path: str) -> int:
    if _bad_int(value):
        raise payload_error(f"{path} must be a non-negative integer")
    return _u64_in_range(value, path)


def _bad_int(value: object) -> bool:
    if isinstance(value, bool):
        return True
    return not isinstance(value, int)


def _u64_in_range(value: int, path: str) -> int:
    if value < 0:
        raise payload_error(f"{path} is outside u64")
    if value > 2**64 - 1:
        raise payload_error(f"{path} is outside u64")
    return value


def as_u32(value: object, path: str) -> int:
    number = as_u64(value, path)
    if number > 2**32 - 1:
        raise payload_error(f"{path} is outside u32")
    return number


def as_u16(value: object, path: str) -> int:
    number = as_u64(value, path)
    if number > 2**16 - 1:
        raise payload_error(f"{path} is outside u16")
    return number


def as_i64(value: object, path: str) -> int:
    if _bad_int(value):
        raise payload_error(f"{path} must be an integer")
    return _i64_in_range(value, path)


def _i64_in_range(value: int, path: str) -> int:
    if value < -(2**63):
        raise payload_error(f"{path} is outside i64")
    if value > 2**63 - 1:
        raise payload_error(f"{path} is outside i64")
    return value


def opt_session(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise payload_error(f"{path} must be a string or null")
    string_bytes(value, path)
    return value


def required_session(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise payload_error(f"{path} must be a string")
    string_bytes(value, path)
    return value
