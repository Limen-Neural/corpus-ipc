"""
Finite f32 conversion and zmij float spelling.
"""

from __future__ import annotations

import json
import math
import struct

from wire_v1.errors import payload_error


def f64_to_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def uint_to_f32(number: int) -> float:
    """Round a non-negative integer to f32, ties-to-even, matching Rust ``as f32``."""
    if number < 0:
        raise ValueError("uint_to_f32 expects a non-negative integer")
    if number == 0:
        return 0.0
    exponent = number.bit_length() - 1
    if exponent >= 128:
        return math.inf
    if exponent <= 23:
        return float(number)
    return _round_wide_uint(number, exponent)


def _round_wide_uint(number: int, exponent: int) -> float:
    shift = exponent - 23
    significand = number >> shift
    if _rounds_up(number, shift, significand):
        significand += 1
    if significand == (1 << 24):
        return _pack_f32(exponent + 1, 0)
    return _pack_f32(exponent, significand & ((1 << 23) - 1))


def _rounds_up(number: int, shift: int, significand: int) -> bool:
    remainder = number & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway:
        return True
    if remainder < halfway:
        return False
    return bool(significand & 1)


def _pack_f32(exponent: int, fraction: int) -> float:
    if exponent >= 128:
        return math.inf
    raw = ((exponent + 127) << 23) | fraction
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def int_to_f32(number: int) -> float:
    if number >= 0:
        return uint_to_f32(number)
    return _negate_f32(uint_to_f32(-number))


def _negate_f32(rounded: float) -> float:
    if rounded == 0.0:
        return -0.0
    return -rounded


def as_f32(value: object, path: str) -> float:
    if _rejects_number(value):
        raise payload_error(f"{path} must be a finite JSON number")
    rounded = _round_number(value, path)
    if not math.isfinite(rounded):
        raise payload_error(f"{path} is non-finite")
    return rounded


def _rejects_number(value: object) -> bool:
    if isinstance(value, bool):
        return True
    return not isinstance(value, (int, float))


def _round_number(value: object, path: str) -> float:
    if isinstance(value, int):
        return _int_number(value, path)
    return _float_number(value, path)


def _int_number(value: int, path: str) -> float:
    if value < -(2**63):
        raise payload_error(f"{path} integer does not fit the wire numeric range")
    if value > (2**64 - 1):
        raise payload_error(f"{path} integer does not fit the wire numeric range")
    return int_to_f32(value)


def _float_number(value: float, path: str) -> float:
    if not math.isfinite(value):
        raise payload_error(f"{path} is non-finite")
    try:
        return f64_to_f32(value)
    except OverflowError:
        raise payload_error(f"{path} is non-finite") from None


def format_f64(value: float) -> str:
    """Format a finite f64 the way serde_json 1.0.149 / zmij prints it."""
    if not math.isfinite(value):
        raise payload_error("non-finite float cannot be encoded")
    if value == 0.0:
        return _signed_zero(value)
    return _format_nonzero(value)


def _signed_zero(value: float) -> str:
    if math.copysign(1.0, value) < 0.0:
        return "-0.0"
    return "0.0"


def _format_nonzero(value: float) -> str:
    negative = math.copysign(1.0, value) < 0.0
    digits, dec_exp = _parse_shortest(json.dumps(abs(value)))
    body = _render_zmij(digits, dec_exp)
    if negative:
        return f"-{body}"
    return body


def _parse_shortest(token: str) -> tuple[str, int]:
    text = token.lower()
    mantissa, exp = _split_exponent(text)
    whole, frac = _split_decimal(mantissa)
    raw = whole + frac
    lead = _leading_zeros(raw)
    digits = raw[lead:]
    dec_exp = exp + len(whole) - 1 - lead
    return digits, dec_exp


def _split_exponent(text: str) -> tuple[str, int]:
    if "e" not in text:
        return text, 0
    mantissa, exp_text = text.split("e")
    return mantissa, int(exp_text)


def _split_decimal(mantissa: str) -> tuple[str, str]:
    if "." not in mantissa:
        return mantissa, ""
    whole, frac = mantissa.split(".")
    return whole, frac


def _leading_zeros(raw: str) -> int:
    lead = 0
    while lead < len(raw) - 1 and raw[lead] == "0":
        lead += 1
    return lead


def _render_zmij(digits: str, dec_exp: int) -> str:
    if -5 <= dec_exp <= 15:
        return _render_fixed(digits, dec_exp)
    return _render_scientific(digits, dec_exp)


def _render_fixed(digits: str, dec_exp: int) -> str:
    if len(digits) - 1 <= dec_exp:
        zeros = dec_exp - (len(digits) - 1)
        return f"{digits}{'0' * zeros}.0"
    if dec_exp >= 0:
        pos = dec_exp + 1
        return f"{digits[:pos]}.{digits[pos:]}"
    zeros = -dec_exp - 1
    return f"0.{'0' * zeros}{digits}"


def _render_scientific(digits: str, dec_exp: int) -> str:
    if len(digits) == 1:
        mantissa = digits
    else:
        mantissa = f"{digits[0]}.{digits[1:]}"
    if dec_exp >= 0:
        sign = "+"
    else:
        sign = "-"
    return f"{mantissa}e{sign}{abs(dec_exp)}"
