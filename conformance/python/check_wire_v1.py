#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline wire-v1 conformance harness (LIM-1327).

Reads ``test-vectors/wire-v1/`` — the hashed golden corpus published with
the canonical encoder — and checks the same outcomes the Rust tests declare:

* SHA-256 of every manifest entry
* positive fixtures decode and re-encode to those exact canonical bytes
* legacy, ``neuron_id`` alias, and integer-to-f32 normalization decode into
  the normalized domain (re-encode is canonical, not byte-identical input)
* negative fixtures fail with the manifest's error class, including
  version checks that run before payload interpretation

Numeric formatting follows serde_json 1.0.149 / zmij f64: finite ``f32``
values are widened with ``f64::from`` and printed in fixed notation for
decimal exponents ``-5..=15``, scientific otherwise, with an explicit
exponent sign. JSON integers that land in ``ConfigValue`` or other ``f32``
fields convert with round-ties-to-even, so ``16777217`` becomes
``16777216.0``.

No third-party packages. Requires CPython >= 3.11.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

CURRENT = 1
MIN_SUPPORTED = 1
LEGACY_UNVERSIONED = 1

MAX_CHANNEL_VALUES = 65_536
MAX_SPIKE_EVENTS = 65_536
MAX_TRACES = 65_536
MAX_GRADIENTS = 65_536
MAX_METADATA_ENTRIES = 256
MAX_STRING_BYTES = 1_024
MAX_AGGREGATE = 131_072

UNIT_VARIANTS = ("Ping", "Shutdown", "TrainingComplete")
KNOWN_VARIANTS = (
    "Spikes",
    "Embeddings",
    "Stimuli",
    "Neuromodulators",
    "Loss",
    "ConfigUpdate",
    "GradientUpdate",
    "EligibilityTraces",
    *UNIT_VARIANTS,
)

EXPECTATIONS = ("decode_ok", "too_old", "too_new", "payload_error", "json_error")
DECODE_OK_KINDS = ("positive", "legacy", "alias", "normalization")


class WireError(Exception):
    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(f"{kind}: {reason}")
        self.kind = kind
        self.reason = reason


def payload_error(reason: str) -> WireError:
    return WireError("payload_error", reason)


def classify_wire_version(version: int) -> str:
    if version < MIN_SUPPORTED:
        return "too_old"
    if version > CURRENT:
        return "too_new"
    return "supported"


# --- f32 / canonical JSON numbers ------------------------------------------


def uint_to_f32(n: int) -> float:
    """Round a non-negative integer to f32, ties-to-even, matching Rust ``as f32``."""
    if n < 0:
        raise ValueError("uint_to_f32 expects a non-negative integer")
    if n == 0:
        return 0.0
    exp = n.bit_length() - 1
    if exp >= 128:
        return math.inf
    if exp <= 23:
        return float(n)
    shift = exp - 23
    significand = n >> shift
    remainder = n & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (significand & 1)):
        significand += 1
        if significand == (1 << 24):
            significand >>= 1
            exp += 1
            if exp >= 128:
                return math.inf
    frac = significand & ((1 << 23) - 1)
    raw = ((exp + 127) << 23) | frac
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def int_to_f32(n: int) -> float:
    if n < 0:
        rounded = uint_to_f32(-n)
        return -rounded if rounded != 0.0 else -0.0
    return uint_to_f32(n)


def f64_to_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def as_f32(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise payload_error(f"{path} must be a finite JSON number")
    if isinstance(value, int):
        if value < -(2**63) or value > (2**64 - 1):
            raise payload_error(f"{path} integer does not fit the wire numeric range")
        rounded = int_to_f32(value)
    else:
        if not math.isfinite(value):
            raise payload_error(f"{path} is non-finite")
        rounded = f64_to_f32(value)
    if not math.isfinite(rounded):
        raise payload_error(f"{path} is non-finite")
    return rounded


def _parse_shortest(token: str) -> tuple[str, int]:
    """Split a positive JSON number into significand digits and decimal exponent."""
    text = token.lower()
    if "e" in text:
        mantissa, exp_text = text.split("e")
        exp = int(exp_text)
    else:
        mantissa, exp = text, 0
    whole, frac = mantissa.split(".") if "." in mantissa else (mantissa, "")
    raw = whole + frac
    lead = 0
    while lead < len(raw) - 1 and raw[lead] == "0":
        lead += 1
    digits = raw[lead:]
    dec_exp = exp + len(whole) - 1 - lead
    return digits, dec_exp


def format_f64(value: float) -> str:
    """zmij ``format_finite`` for f64, which is how canonical wire-v1 prints floats."""
    if not math.isfinite(value):
        raise payload_error("non-finite float cannot be encoded")
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0.0 else "0.0"
    negative = math.copysign(1.0, value) < 0.0
    digits, dec_exp = _parse_shortest(json.dumps(abs(value)))
    body = _render_zmij(digits, dec_exp)
    return f"-{body}" if negative else body


def _render_zmij(digits: str, dec_exp: int) -> str:
    length = len(digits)
    if -5 <= dec_exp <= 15:
        if length - 1 <= dec_exp:
            zeros = dec_exp - (length - 1)
            return f"{digits}{'0' * zeros}.0"
        if dec_exp >= 0:
            pos = dec_exp + 1
            return f"{digits[:pos]}.{digits[pos:]}"
        zeros = -dec_exp - 1
        return f"0.{'0' * zeros}{digits}"
    mantissa = digits if length == 1 else f"{digits[0]}.{digits[1:]}"
    sign = "+" if dec_exp >= 0 else "-"
    exp = abs(dec_exp)
    width = 2 if exp < 100 else 3
    return f"{mantissa}e{sign}{exp:0{width}d}"


def dumps(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_f64(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(dumps(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
        body = ",".join(
            dumps(key) + ":" + dumps(item) for key, item in items
        )
        return "{" + body + "}"
    raise TypeError(f"cannot encode {type(value).__name__}")


# --- model -----------------------------------------------------------------


@dataclass
class SpikeEvent:
    channel: int
    time: int
    strength: float


@dataclass
class BatchMetadata:
    processing_latency_ns: int | None
    source: str | None
    custom: dict[str, str]


@dataclass
class SpikeBatch:
    session_id: str | None
    batch_id: int
    timestamp: int
    spikes: list[SpikeEvent]
    metadata: BatchMetadata | None


@dataclass
class EmbeddingBatch:
    session_id: str | None
    batch_id: int
    embedding: list[float]
    sequence_length: int


@dataclass
class StimulusBatch:
    session_id: str | None
    batch_id: int
    timestamp: int
    values: list[float]
    valid_mask: list[bool] | None
    metadata: BatchMetadata | None


@dataclass
class NeuromodulatorSnapshot:
    tick: int
    dopamine: float
    cortisol: float
    acetylcholine: float
    tempo: float


@dataclass
class ConfigValue:
    kind: str
    value: object


@dataclass
class ConfigPayload:
    session_id: str | None
    config: dict[str, ConfigValue]


@dataclass
class GradientUpdate:
    layer_id: str
    gradients: list[float]
    eligibility_trace: list[float] | None


@dataclass
class GradientBatch:
    session_id: str
    batch_id: int
    gradients: list[GradientUpdate]


@dataclass
class TraceData:
    channel_id: int
    trace_value: float
    last_spike_time: int


@dataclass
class TraceBatch:
    session_id: str
    batch_id: int
    traces: list[TraceData]


@dataclass
class Message:
    variant: str
    body: object


# --- validation helpers ----------------------------------------------------


def _string_bytes(value: str, path: str) -> None:
    if len(value.encode("utf-8")) > MAX_STRING_BYTES:
        raise payload_error(f"{path} exceeds {MAX_STRING_BYTES} bytes")


def _opt_string(value: str | None, path: str) -> None:
    if value is not None:
        _string_bytes(value, path)


def _count(actual: int, limit: int, path: str) -> None:
    if actual > limit:
        raise payload_error(f"{path} length {actual} exceeds {limit}")


def _finite(value: float, path: str) -> None:
    if not math.isfinite(value):
        raise payload_error(f"{path} is non-finite")


def _range(value: float, low: float, high: float, path: str) -> None:
    _finite(value, path)
    if value < low or value > high:
        raise payload_error(f"{path} {format_f64(value)} is outside [{low}, {high}]")


def _add(total: int, count: int, path: str) -> int:
    total += count
    if total > MAX_AGGREGATE:
        raise payload_error(f"{path} aggregate exceeds {MAX_AGGREGATE}")
    return total


def _require_object(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        raise payload_error(f"{path} must be a JSON object")
    return value


def _require(obj: dict, key: str, path: str) -> object:
    if key not in obj:
        raise payload_error(f"missing {path}")
    return obj[key]


def _as_u64(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise payload_error(f"{path} must be a non-negative integer")
    if value < 0 or value > 2**64 - 1:
        raise payload_error(f"{path} is outside u64")
    return value


def _as_u32(value: object, path: str) -> int:
    number = _as_u64(value, path)
    if number > 2**32 - 1:
        raise payload_error(f"{path} is outside u32")
    return number


def _as_u16(value: object, path: str) -> int:
    number = _as_u64(value, path)
    if number > 2**16 - 1:
        raise payload_error(f"{path} is outside u16")
    return number


def _as_i64(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise payload_error(f"{path} must be an integer")
    if value < -(2**63) or value > 2**63 - 1:
        raise payload_error(f"{path} is outside i64")
    return value


def _as_usize(value: object, path: str) -> int:
    return _as_u64(value, path)


def _opt_session(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise payload_error(f"{path} must be a string or null")
    _string_bytes(value, path)
    return value


def _required_session(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise payload_error(f"{path} must be a string")
    _string_bytes(value, path)
    return value


def _f32_list(value: object, path: str, limit: int) -> list[float]:
    if not isinstance(value, list):
        raise payload_error(f"{path} must be an array")
    _count(len(value), limit, path)
    return [as_f32(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _decode_metadata(value: object, path: str) -> BatchMetadata:
    obj = _require_object(value, path)
    latency = _require(obj, "processing_latency_ns", f"{path}.processing_latency_ns")
    if latency is not None:
        latency = _as_u64(latency, f"{path}.processing_latency_ns")
    if "source" in obj:
        source = _opt_session(obj["source"], f"{path}.source")
    else:
        source = None
    if "custom" in obj:
        custom_raw = obj["custom"]
        if not isinstance(custom_raw, dict):
            raise payload_error(f"{path}.custom must be an object")
        _count(len(custom_raw), MAX_METADATA_ENTRIES, f"{path}.custom")
        custom: dict[str, str] = {}
        for key, item in custom_raw.items():
            if key == "":
                raise payload_error(f"{path}.custom.<empty> is not a valid key")
            _string_bytes(key, f"{path}.custom.<key>")
            if not isinstance(item, str):
                raise payload_error(f"{path}.custom.{key} must be a string")
            _string_bytes(item, f"{path}.custom.<value>")
            custom[key] = item
    else:
        custom = {}
    _count(len(custom), MAX_METADATA_ENTRIES, f"{path}.custom")
    return BatchMetadata(latency, source, custom)


def _validate_metadata(metadata: BatchMetadata | None, already: int, path: str) -> None:
    total = _add(0, already, path)
    if metadata is None:
        return
    _opt_string(metadata.source, f"{path}.source")
    _count(len(metadata.custom), MAX_METADATA_ENTRIES, f"{path}.custom")
    for key, item in metadata.custom.items():
        if key == "":
            raise payload_error(f"{path}.custom.<empty> is not a valid key")
        _string_bytes(key, f"{path}.custom.<key>")
        _string_bytes(item, f"{path}.custom.<value>")
    _add(total, len(metadata.custom), path)


def _decode_spike(value: object, path: str) -> SpikeEvent:
    obj = _require_object(value, path)
    event = SpikeEvent(
        _as_u16(_require(obj, "channel", f"{path}.channel"), f"{path}.channel"),
        _as_u32(_require(obj, "time", f"{path}.time"), f"{path}.time"),
        as_f32(_require(obj, "strength", f"{path}.strength"), f"{path}.strength"),
    )
    _finite(event.strength, f"{path}.strength")
    return event


def decode_spikes(value: object) -> SpikeBatch:
    obj = _require_object(value, "Spikes")
    session = (
        _opt_session(obj["session_id"], "session_id") if "session_id" in obj else None
    )
    spikes_raw = _require(obj, "spikes", "spikes")
    if not isinstance(spikes_raw, list):
        raise payload_error("spikes must be an array")
    _count(len(spikes_raw), MAX_SPIKE_EVENTS, "spikes")
    spikes = [_decode_spike(item, f"spikes[{index}]") for index, item in enumerate(spikes_raw)]
    if "metadata" not in obj:
        raise payload_error("missing metadata")
    metadata = obj["metadata"]
    decoded_meta = None if metadata is None else _decode_metadata(metadata, "metadata")
    batch = SpikeBatch(
        session,
        _as_u64(_require(obj, "batch_id", "batch_id"), "batch_id"),
        _as_u64(_require(obj, "timestamp", "timestamp"), "timestamp"),
        spikes,
        decoded_meta,
    )
    _opt_string(batch.session_id, "session_id")
    _validate_metadata(batch.metadata, len(batch.spikes), "spikes")
    return batch


def decode_embeddings(value: object) -> EmbeddingBatch:
    obj = _require_object(value, "Embeddings")
    embedding = _f32_list(
        _require(obj, "embedding", "embedding"), "embedding", MAX_CHANNEL_VALUES
    )
    sequence = _as_usize(
        _require(obj, "sequence_length", "sequence_length"), "sequence_length"
    )
    _count(sequence, MAX_AGGREGATE, "sequence_length")
    for index, item in enumerate(embedding):
        _finite(item, f"embedding[{index}]")
    _add(0, len(embedding), "embedding")
    return EmbeddingBatch(
        _opt_session(_require(obj, "session_id", "session_id"), "session_id"),
        _as_u64(_require(obj, "batch_id", "batch_id"), "batch_id"),
        embedding,
        sequence,
    )


def decode_stimuli(value: object) -> StimulusBatch:
    obj = _require_object(value, "Stimuli")
    values = _f32_list(_require(obj, "values", "values"), "values", MAX_CHANNEL_VALUES)
    for index, item in enumerate(values):
        _finite(item, f"values[{index}]")
    mask_raw = _require(obj, "valid_mask", "valid_mask")
    if mask_raw is None:
        mask = None
    elif isinstance(mask_raw, list):
        _count(len(mask_raw), MAX_CHANNEL_VALUES, "valid_mask")
        mask = []
        for index, item in enumerate(mask_raw):
            if not isinstance(item, bool):
                raise payload_error(f"valid_mask[{index}] must be a boolean")
            mask.append(item)
        if len(mask) != len(values):
            raise payload_error(
                f"valid_mask length {len(mask)} does not match values length {len(values)}"
            )
    else:
        raise payload_error("valid_mask must be an array or null")
    if "metadata" not in obj:
        raise payload_error("missing metadata")
    metadata = obj["metadata"]
    decoded_meta = None if metadata is None else _decode_metadata(metadata, "metadata")
    _validate_metadata(decoded_meta, len(values), "values")
    return StimulusBatch(
        _opt_session(_require(obj, "session_id", "session_id"), "session_id"),
        _as_u64(_require(obj, "batch_id", "batch_id"), "batch_id"),
        _as_u64(_require(obj, "timestamp", "timestamp"), "timestamp"),
        values,
        mask,
        decoded_meta,
    )


def decode_neuromodulators(value: object) -> NeuromodulatorSnapshot:
    obj = _require_object(value, "Neuromodulators")
    snapshot = NeuromodulatorSnapshot(
        _as_i64(_require(obj, "tick", "tick"), "tick"),
        as_f32(_require(obj, "dopamine", "dopamine"), "dopamine"),
        as_f32(_require(obj, "cortisol", "cortisol"), "cortisol"),
        as_f32(_require(obj, "acetylcholine", "acetylcholine"), "acetylcholine"),
        as_f32(_require(obj, "tempo", "tempo"), "tempo"),
    )
    _range(snapshot.dopamine, 0.0, 1.0, "dopamine")
    _range(snapshot.cortisol, 0.0, 1.0, "cortisol")
    _range(snapshot.acetylcholine, 0.0, 1.0, "acetylcholine")
    _range(snapshot.tempo, 0.5, 2.0, "tempo")
    return snapshot


def decode_loss(value: object) -> float:
    loss = as_f32(value, "Loss")
    _finite(loss, "Loss")
    return loss


def _decode_config_value(value: object, path: str) -> ConfigValue:
    if isinstance(value, bool):
        return ConfigValue("bool", value)
    if isinstance(value, str):
        _string_bytes(value, path)
        return ConfigValue("string", value)
    if isinstance(value, list):
        numbers = _f32_list(value, path, MAX_CHANNEL_VALUES)
        for index, item in enumerate(numbers):
            _finite(item, f"{path}[{index}]")
        return ConfigValue("array", numbers)
    if isinstance(value, dict):
        raise payload_error(f"{path} must be a number, string, boolean, or array of floats")
    number = as_f32(value, path)
    _finite(number, path)
    return ConfigValue("float", number)


def decode_config(value: object) -> ConfigPayload:
    obj = _require_object(value, "ConfigUpdate")
    raw = _require(obj, "config", "config")
    if not isinstance(raw, dict):
        raise payload_error("config must be an object")
    _count(len(raw), MAX_METADATA_ENTRIES, "config")
    config: dict[str, ConfigValue] = {}
    total = _add(0, len(raw), "config")
    for key, item in raw.items():
        if key == "":
            raise payload_error("config.<empty> is not a valid key")
        _string_bytes(key, "config.<key>")
        decoded = _decode_config_value(item, f"config.{key}")
        config[key] = decoded
        if decoded.kind == "array":
            if not isinstance(decoded.value, list):
                raise payload_error(f"config.{key} array was not decoded")
            total = _add(total, len(decoded.value), "config")
    return ConfigPayload(
        _opt_session(_require(obj, "session_id", "session_id"), "session_id"),
        config,
    )


def _decode_gradient_row(value: object, path: str) -> GradientUpdate:
    obj = _require_object(value, path)
    gradients = _f32_list(
        _require(obj, "gradients", f"{path}.gradients"),
        f"{path}.gradients",
        MAX_CHANNEL_VALUES,
    )
    for index, item in enumerate(gradients):
        _finite(item, f"{path}.gradients[{index}]")
    if "eligibility_trace" in obj and obj["eligibility_trace"] is not None:
        trace = _f32_list(
            obj["eligibility_trace"],
            f"{path}.eligibility_trace",
            MAX_CHANNEL_VALUES,
        )
        for index, item in enumerate(trace):
            _finite(item, f"{path}.eligibility_trace[{index}]")
    else:
        trace = None
    layer = _required_session(_require(obj, "layer_id", f"{path}.layer_id"), f"{path}.layer_id")
    return GradientUpdate(layer, gradients, trace)


def decode_gradients(value: object) -> GradientBatch:
    obj = _require_object(value, "GradientUpdate")
    raw = _require(obj, "gradients", "gradients")
    if not isinstance(raw, list):
        raise payload_error("gradients must be an array")
    _count(len(raw), MAX_GRADIENTS, "gradients")
    rows = [_decode_gradient_row(item, f"gradients[{index}]") for index, item in enumerate(raw)]
    seen: set[str] = set()
    total = _add(0, len(rows), "gradients")
    for row in rows:
        if row.layer_id in seen:
            raise payload_error(f"duplicate layer_id {row.layer_id!r}")
        seen.add(row.layer_id)
        total = _add(total, len(row.gradients), "gradients")
        if row.eligibility_trace is not None:
            total = _add(total, len(row.eligibility_trace), "gradients")
    return GradientBatch(
        _required_session(_require(obj, "session_id", "session_id"), "session_id"),
        _as_u64(_require(obj, "batch_id", "batch_id"), "batch_id"),
        rows,
    )


def _decode_trace(value: object, path: str) -> TraceData:
    obj = _require_object(value, path)
    has_channel = "channel_id" in obj
    has_alias = "neuron_id" in obj
    if has_channel and has_alias:
        raise payload_error(f"{path} has both channel_id and neuron_id")
    if has_channel:
        channel = _as_u16(obj["channel_id"], f"{path}.channel_id")
    elif has_alias:
        channel = _as_u16(obj["neuron_id"], f"{path}.neuron_id")
    else:
        raise payload_error(f"missing {path}.channel_id")
    data = TraceData(
        channel,
        as_f32(_require(obj, "trace_value", f"{path}.trace_value"), f"{path}.trace_value"),
        _as_u32(
            _require(obj, "last_spike_time", f"{path}.last_spike_time"),
            f"{path}.last_spike_time",
        ),
    )
    _finite(data.trace_value, f"{path}.trace_value")
    return data


def decode_traces(value: object) -> TraceBatch:
    obj = _require_object(value, "EligibilityTraces")
    raw = _require(obj, "traces", "traces")
    if not isinstance(raw, list):
        raise payload_error("traces must be an array")
    _count(len(raw), MAX_TRACES, "traces")
    traces = [_decode_trace(item, f"traces[{index}]") for index, item in enumerate(raw)]
    seen: set[int] = set()
    for trace in traces:
        if trace.channel_id in seen:
            raise payload_error(f"duplicate channel_id {trace.channel_id}")
        seen.add(trace.channel_id)
    _add(0, len(traces), "traces")
    return TraceBatch(
        _required_session(_require(obj, "session_id", "session_id"), "session_id"),
        _as_u64(_require(obj, "batch_id", "batch_id"), "batch_id"),
        traces,
    )


DECODERS = {
    "Spikes": decode_spikes,
    "Embeddings": decode_embeddings,
    "Stimuli": decode_stimuli,
    "Neuromodulators": decode_neuromodulators,
    "Loss": decode_loss,
    "ConfigUpdate": decode_config,
    "GradientUpdate": decode_gradients,
    "EligibilityTraces": decode_traces,
}


def decode_payload(value: object) -> Message:
    if isinstance(value, str):
        if value in UNIT_VARIANTS:
            return Message(value, None)
        raise payload_error(f"unknown IpcMessage variant {value!r}")
    if not isinstance(value, dict):
        raise WireError(
            "not_an_object",
            "IpcMessage must be an externally tagged object or a unit-variant string",
        )
    if "type" in value and set(value) != {"type"}:
        raise payload_error(
            "internally tagged message is not supported "
            f"(found type={value.get('type')!r}); expected an externally tagged IpcMessage"
        )
    if len(value) != 1:
        raise payload_error(
            "externally tagged IpcMessage must have exactly one variant key, "
            f"found {sorted(value)}"
        )
    name, body = next(iter(value.items()))
    if name in UNIT_VARIANTS:
        if body not in (None, {}):
            raise payload_error(f"{name} is a unit variant and must be a JSON string")
        return Message(name, None)
    decoder = DECODERS.get(name)
    if decoder is None:
        raise payload_error(f"unknown IpcMessage variant {name!r}")
    return Message(name, decoder(body))


def _parse_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireError(
            "invalid_version",
            f"wire_version must be a non-negative integer fitting u32, got {value!r}",
        )
    if value < 0 or value > 2**32 - 1:
        raise WireError(
            "invalid_version",
            f"wire_version must be a non-negative integer fitting u32, got {value}",
        )
    return value


def decode_message(data: bytes) -> Message:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WireError("json_error", f"envelope JSON error: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireError("json_error", f"envelope JSON error: {exc}") from exc
    if isinstance(value, dict) and "wire_version" in value:
        version = _parse_version(value["wire_version"])
        kind = classify_wire_version(version)
        if kind == "too_old":
            raise WireError(
                "too_old",
                f"wire version {version} is too old (minimum supported is {MIN_SUPPORTED}); "
                "payload was not interpreted",
            )
        if kind == "too_new":
            raise WireError(
                "too_new",
                f"wire version {version} is too new (current is {CURRENT}); "
                "payload was not interpreted",
            )
        if "payload" not in value:
            raise WireError("missing_payload", "missing payload on compatibility envelope")
        return decode_payload(value["payload"])
    if classify_wire_version(LEGACY_UNVERSIONED) != "supported":
        raise WireError("too_old", "legacy unversioned JSON is outside the supported window")
    return decode_payload(value)


# --- canonical encode ------------------------------------------------------


def _encode_metadata(metadata: BatchMetadata | None) -> object:
    if metadata is None:
        return None
    return {
        "processing_latency_ns": metadata.processing_latency_ns,
        "source": metadata.source,
        "custom": metadata.custom,
    }


def _encode_config_value(value: ConfigValue) -> object:
    if value.kind == "float":
        return value.value
    if value.kind == "array":
        return value.value
    if value.kind in ("bool", "string"):
        return value.value
    raise payload_error(f"cannot encode config value {value.kind}")


def encode_payload(message: Message) -> object:
    if message.variant in UNIT_VARIANTS:
        return message.variant
    body = message.body
    if message.variant == "Loss":
        return {"Loss": body}
    if message.variant == "Spikes":
        assert isinstance(body, SpikeBatch)
        return {
            "Spikes": {
                "session_id": body.session_id,
                "batch_id": body.batch_id,
                "timestamp": body.timestamp,
                "spikes": [
                    {"channel": spike.channel, "time": spike.time, "strength": spike.strength}
                    for spike in body.spikes
                ],
                "metadata": _encode_metadata(body.metadata),
            }
        }
    if message.variant == "Embeddings":
        assert isinstance(body, EmbeddingBatch)
        return {
            "Embeddings": {
                "session_id": body.session_id,
                "batch_id": body.batch_id,
                "embedding": body.embedding,
                "sequence_length": body.sequence_length,
            }
        }
    if message.variant == "Stimuli":
        assert isinstance(body, StimulusBatch)
        return {
            "Stimuli": {
                "session_id": body.session_id,
                "batch_id": body.batch_id,
                "timestamp": body.timestamp,
                "values": body.values,
                "valid_mask": body.valid_mask,
                "metadata": _encode_metadata(body.metadata),
            }
        }
    if message.variant == "Neuromodulators":
        assert isinstance(body, NeuromodulatorSnapshot)
        return {
            "Neuromodulators": {
                "tick": body.tick,
                "dopamine": body.dopamine,
                "cortisol": body.cortisol,
                "acetylcholine": body.acetylcholine,
                "tempo": body.tempo,
            }
        }
    if message.variant == "ConfigUpdate":
        assert isinstance(body, ConfigPayload)
        return {
            "ConfigUpdate": {
                "session_id": body.session_id,
                "config": {
                    key: _encode_config_value(item) for key, item in body.config.items()
                },
            }
        }
    if message.variant == "GradientUpdate":
        assert isinstance(body, GradientBatch)
        return {
            "GradientUpdate": {
                "session_id": body.session_id,
                "batch_id": body.batch_id,
                "gradients": [
                    {
                        "layer_id": row.layer_id,
                        "gradients": row.gradients,
                        "eligibility_trace": row.eligibility_trace,
                    }
                    for row in body.gradients
                ],
            }
        }
    if message.variant == "EligibilityTraces":
        assert isinstance(body, TraceBatch)
        return {
            "EligibilityTraces": {
                "session_id": body.session_id,
                "batch_id": body.batch_id,
                "traces": [
                    {
                        "channel_id": trace.channel_id,
                        "trace_value": trace.trace_value,
                        "last_spike_time": trace.last_spike_time,
                    }
                    for trace in body.traces
                ],
            }
        }
    raise payload_error(f"cannot encode variant {message.variant}")


def encode_canonical(message: Message) -> bytes:
    text = dumps({"payload": encode_payload(message), "wire_version": CURRENT})
    return text.encode("utf-8")


# --- corpus checks ---------------------------------------------------------


def vector_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "test-vectors" / "wire-v1"


def load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise SystemExit("manifest schema_version must be 1")
    if "wire v1" not in manifest.get("encoding_profile", ""):
        raise SystemExit("manifest encoding_profile must name wire v1")
    return manifest


def check_manifest_hashes(root: Path, manifest: dict) -> list[dict]:
    fixtures = manifest["fixtures"]
    seen: set[str] = set()
    for entry in fixtures:
        name = entry["name"]
        if name in seen:
            raise SystemExit(f"duplicate fixture name {name}")
        seen.add(name)
        if entry["expect"] not in EXPECTATIONS:
            raise SystemExit(f"{name}: unknown expectation {entry['expect']}")
        if entry["kind"] == "positive" and entry.get("variant") not in KNOWN_VARIANTS:
            raise SystemExit(f"{name}: unknown positive variant {entry.get('variant')}")
        blob = (root / entry["file"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"sha256 drift on {name}: {digest} != {entry['sha256']}")
    covered = {entry["variant"] for entry in fixtures if entry["kind"] == "positive"}
    missing = [variant for variant in KNOWN_VARIANTS if variant not in covered]
    if missing:
        raise SystemExit(f"missing positive vector for {missing}")
    return fixtures


def _fail(name: str, detail: str) -> None:
    raise SystemExit(f"FAIL {name}: {detail}")


def check_fixture(root: Path, entry: dict) -> None:
    name = entry["name"]
    blob = (root / entry["file"]).read_bytes()
    expect = entry["expect"]
    try:
        message = decode_message(blob)
    except WireError as exc:
        if expect == "decode_ok":
            _fail(name, f"expected decode_ok, got {exc.kind}: {exc.reason}")
        if exc.kind != expect:
            _fail(name, f"expected {expect}, got {exc.kind}: {exc.reason}")
        if not exc.reason.strip():
            _fail(name, "rejection reason is empty")
        return
    if expect != "decode_ok":
        _fail(name, f"expected {expect}, but decode succeeded as {message.variant}")
    if entry["kind"] not in DECODE_OK_KINDS:
        _fail(name, f"unexpected kind {entry['kind']}")
    if entry["kind"] == "positive":
        if message.variant != entry["variant"]:
            _fail(name, f"variant {message.variant} != {entry['variant']}")
        encoded = encode_canonical(message)
        if encoded != blob:
            _fail(
                name,
                "canonical re-encode drifted\n"
                f"  got {encoded.decode()}\n"
                f"  want {blob.decode()}",
            )
        return
    encoded = encode_canonical(message)
    again = decode_message(encoded)
    if encode_canonical(again) != encoded:
        _fail(name, "normalized re-encode is not stable")
    if encoded == blob:
        _fail(name, "hand-maintained input was already canonical; check the fixture kind")


def check_named_cases(root: Path, fixtures: list[dict]) -> None:
    by_name = {entry["name"]: entry for entry in fixtures}
    if classify_wire_version(0) != "too_old":
        _fail("classify", "version 0 must be too_old")
    if classify_wire_version(1) != "supported":
        _fail("classify", "version 1 must be supported")
    if classify_wire_version(2) != "too_new":
        _fail("classify", "version 2 must be too_new")

    too_new = decode_error(root, by_name["v2_too_new_malformed_payload"])
    if too_new.kind != "too_new" or "payload was not interpreted" not in too_new.reason:
        _fail("v2_too_new_malformed_payload", f"version must win, got {too_new}")
    if "2" not in too_new.reason:
        _fail("v2_too_new_malformed_payload", "reason must name the found version")

    unknown = decode_error(root, by_name["v1_unknown_variant"])
    if unknown.kind != "payload_error" or "NotARealVariant" not in unknown.reason:
        _fail("v1_unknown_variant", f"unexpected rejection {unknown}")

    wrong = decode_error(root, by_name["v1_wrong_tagging"])
    if wrong.kind != "payload_error" or "internally tagged" not in wrong.reason:
        _fail("v1_wrong_tagging", f"unexpected rejection {wrong}")

    internal = (
        b'{"wire_version":1,"payload":{"type":"Stimuli","values":[0.0],"valid_mask":null}}'
    )
    try:
        decode_message(internal)
    except WireError as exc:
        if exc.kind != "payload_error" or "internally tagged" not in exc.reason:
            _fail("internal_stimuli", f"unexpected rejection {exc}")
    else:
        _fail("internal_stimuli", "internally tagged Stimuli was accepted")

    alias_bytes = (root / by_name["alias_neuron_id_traces"]["file"]).read_bytes()
    if b"neuron_id" not in alias_bytes or b"channel_id" in alias_bytes:
        _fail("alias_neuron_id_traces", "fixture must use the neuron_id alias only")
    alias = decode_message(alias_bytes)
    if not isinstance(alias.body, TraceBatch) or alias.body.traces[0].channel_id != 5:
        _fail("alias_neuron_id_traces", "neuron_id did not decode as channel_id 5")
    alias_canonical = encode_canonical(alias)
    if b"neuron_id" in alias_canonical or b'"channel_id":5' not in alias_canonical:
        _fail("alias_neuron_id_traces", "canonical output must use channel_id")

    norm_bytes = (root / by_name["normalization_integer_config"]["file"]).read_bytes()
    if b"16777217" not in norm_bytes:
        _fail("normalization_integer_config", "fixture must keep the literal 2^24+1 input")
    norm = decode_message(norm_bytes)
    if not isinstance(norm.body, ConfigPayload):
        _fail("normalization_integer_config", "expected ConfigUpdate")
    big = norm.body.config["big"]
    small = norm.body.config["n"]
    if big.kind != "float" or small.kind != "float":
        _fail("normalization_integer_config", "JSON integers must decode as floats")
    if big.value != 16_777_216.0 or int(big.value) == 16_777_217:
        _fail("normalization_integer_config", f"big decoded as {big.value!r}")
    if small.value != 42.0:
        _fail("normalization_integer_config", f"n decoded as {small.value!r}")
    norm_text = encode_canonical(norm).decode()
    if "16777217" in norm_text or '"big":16777216.0' not in norm_text:
        _fail("normalization_integer_config", f"canonical floats drifted: {norm_text}")
    if '"n":42.0' not in norm_text:
        _fail("normalization_integer_config", f"42 must be a float: {norm_text}")

    signed = decode_message((root / by_name["signed_zero"]["file"]).read_bytes())
    if not isinstance(signed.body, ConfigPayload):
        _fail("signed_zero", "expected ConfigUpdate")
    neg = signed.body.config["neg"].value
    pos = signed.body.config["pos"].value
    if math.copysign(1.0, neg) != -1.0 or math.copysign(1.0, pos) != 1.0:
        _fail("signed_zero", f"signs were {neg!r} {pos!r}")

    masked = decode_message((root / by_name["stimuli_masked"]["file"]).read_bytes())
    unmasked = decode_message((root / by_name["stimuli_unmasked"]["file"]).read_bytes())
    if not isinstance(masked.body, StimulusBatch) or masked.body.valid_mask != [
        True,
        False,
        True,
    ]:
        _fail("stimuli_masked", "valid_mask did not round-trip")
    if masked.body.values[1] != 0.0:
        _fail("stimuli_masked", "masked placeholder value changed")
    if not isinstance(unmasked.body, StimulusBatch) or unmasked.body.valid_mask is not None:
        _fail("stimuli_unmasked", "null valid_mask must decode as absent")
    mask_err = decode_error(root, by_name["v1_invalid_stimulus_mask"])
    if mask_err.kind != "payload_error" or "valid_mask" not in mask_err.reason:
        _fail("v1_invalid_stimulus_mask", f"unexpected rejection {mask_err}")

    neuro_err = decode_error(root, by_name["v1_invalid_neuromod_range"])
    if neuro_err.kind != "payload_error" or "dopamine" not in neuro_err.reason:
        _fail("v1_invalid_neuromod_range", f"unexpected rejection {neuro_err}")

    legacy_ping = decode_message(
        (root / by_name["legacy_unversioned_ping"]["file"]).read_bytes()
    )
    if legacy_ping.variant != "Ping":
        _fail("legacy_unversioned_ping", "expected Ping")
    ping_bytes = (root / by_name["ping"]["file"]).read_bytes()
    if encode_canonical(legacy_ping) != ping_bytes:
        _fail("legacy_unversioned_ping", "normalized Ping must match canonical ping.json")

    legacy_spikes = decode_message(
        (root / by_name["legacy_unversioned_spikes"]["file"]).read_bytes()
    )
    if not isinstance(legacy_spikes.body, SpikeBatch):
        _fail("legacy_unversioned_spikes", "expected Spikes")
    batch = legacy_spikes.body
    if (
        batch.session_id != "sess-1"
        or batch.batch_id != 7
        or batch.metadata is not None
        or len(batch.spikes) != 1
        or batch.spikes[0].channel != 3
    ):
        _fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    if not encode_canonical(legacy_spikes).startswith(b'{"payload":'):
        _fail("legacy_unversioned_spikes", "normalized form must be a v1 envelope")


def decode_error(root: Path, entry: dict) -> WireError:
    try:
        decode_message((root / entry["file"]).read_bytes())
    except WireError as exc:
        return exc
    _fail(entry["name"], "expected a rejection")
    raise AssertionError("unreachable")


def check_numeric_contract() -> None:
    """Lock f32 rounding and zmij notation even for values absent from the corpus."""
    if uint_to_f32(16_777_217) != 16_777_216.0:
        raise SystemExit("16777217 must round to f32 16777216")
    f32_max = f64_to_f32(3.4028234663852886e38)
    expected = [
        (1e-6, "1e-06"),
        (1e-5, "0.00001"),
        (1e-4, "0.0001"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (16_777_216.0, "16777216.0"),
        (42.0, "42.0"),
        (-0.0, "-0.0"),
        (f32_max, "3.4028234663852886e+38"),
        (f64_to_f32(0.8), "0.800000011920929"),
    ]
    for value, text in expected:
        got = format_f64(value)
        if got != text:
            raise SystemExit(f"float format {value!r}: got {got}, want {text}")


def main() -> int:
    check_numeric_contract()
    root = vector_dir()
    manifest = load_manifest(root)
    fixtures = check_manifest_hashes(root, manifest)
    for entry in fixtures:
        check_fixture(root, entry)
    check_named_cases(root, fixtures)
    print(f"ok {len(fixtures)} wire-v1 fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
