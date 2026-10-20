"""Decoders for config, gradient, trace, and loss payloads."""

from __future__ import annotations

from wire_v1.decode_input import check_finite_list, read_f32_list
from wire_v1.errors import payload_error
from wire_v1.limits import MAX_CHANNEL_VALUES, MAX_GRADIENTS, MAX_METADATA_ENTRIES, MAX_TRACES
from wire_v1.model import ConfigPayload, ConfigValue, GradientBatch, GradientUpdate, TraceBatch, TraceData
from wire_v1.numbers import as_f32
from wire_v1.scalars import (
    add_total,
    as_u16,
    as_u32,
    as_u64,
    check_count,
    check_finite,
    opt_session,
    require_key,
    require_object,
    required_session,
    string_bytes,
)


def decode_loss(value: object) -> float:
    loss = as_f32(value, "Loss")
    check_finite(loss, "Loss")
    return loss


def decode_config(value: object) -> ConfigPayload:
    obj = require_object(value, "ConfigUpdate")
    raw = require_object(require_key(obj, "config", "config"), "config")
    check_count(len(raw), MAX_METADATA_ENTRIES, "config")
    config, total = _config_entries(raw)
    add_total(total, 0, "config")
    return ConfigPayload(opt_session(require_key(obj, "session_id", "session_id"), "session_id"), config)


def _config_entries(raw: dict) -> tuple[dict[str, ConfigValue], int]:
    config: dict[str, ConfigValue] = {}
    total = add_total(0, len(raw), "config")
    for key, item in raw.items():
        decoded = _one_config(key, item)
        config[key] = decoded
        total = _count_array(total, decoded)
    return config, total


def _one_config(key: str, item: object) -> ConfigValue:
    if key == "":
        raise payload_error("config.<empty> is not a valid key")
    string_bytes(key, "config.<key>")
    return decode_config_value(item, f"config.{key}")


def _count_array(total: int, decoded: ConfigValue) -> int:
    if decoded.kind != "array":
        return total
    if not isinstance(decoded.value, list):
        raise payload_error("config array was not decoded")
    return add_total(total, len(decoded.value), "config")


def decode_config_value(value: object, path: str) -> ConfigValue:
    if isinstance(value, bool):
        return ConfigValue("bool", value)
    if isinstance(value, str):
        return _config_string(value, path)
    if isinstance(value, list):
        return _config_array(value, path)
    if isinstance(value, dict):
        raise payload_error(f"{path} must be a number, string, boolean, or array of floats")
    return _config_float(value, path)


def _config_string(value: str, path: str) -> ConfigValue:
    string_bytes(value, path)
    return ConfigValue("string", value)


def _config_array(value: list, path: str) -> ConfigValue:
    numbers = read_f32_list(value, path, MAX_CHANNEL_VALUES)
    check_finite_list(numbers, path)
    return ConfigValue("array", numbers)


def _config_float(value: object, path: str) -> ConfigValue:
    number = as_f32(value, path)
    check_finite(number, path)
    return ConfigValue("float", number)


def decode_gradients(value: object) -> GradientBatch:
    obj = require_object(value, "GradientUpdate")
    rows = read_gradient_rows(require_key(obj, "gradients", "gradients"))
    return GradientBatch(
        required_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        rows,
    )


def read_gradient_rows(value: object) -> list[GradientUpdate]:
    if not isinstance(value, list):
        raise payload_error("gradients must be an array")
    check_count(len(value), MAX_GRADIENTS, "gradients")
    rows = [decode_gradient_row(item, index) for index, item in enumerate(value)]
    _unique_layers(rows)
    return rows


def _unique_layers(rows: list[GradientUpdate]) -> None:
    seen: set[str] = set()
    total = add_total(0, len(rows), "gradients")
    for row in rows:
        total = _one_layer(row, seen, total)
    add_total(total, 0, "gradients")


def _one_layer(row: GradientUpdate, seen: set[str], total: int) -> int:
    if row.layer_id in seen:
        raise payload_error(f"duplicate layer_id {row.layer_id!r}")
    seen.add(row.layer_id)
    total = add_total(total, len(row.gradients), "gradients")
    if row.eligibility_trace is None:
        return total
    return add_total(total, len(row.eligibility_trace), "gradients")


def decode_gradient_row(value: object, index: int) -> GradientUpdate:
    path = f"gradients[{index}]"
    obj = require_object(value, path)
    gradients = read_f32_list(require_key(obj, "gradients", f"{path}.gradients"), f"{path}.gradients", MAX_CHANNEL_VALUES)
    check_finite_list(gradients, f"{path}.gradients")
    trace = read_eligibility(obj, path)
    layer = required_session(require_key(obj, "layer_id", f"{path}.layer_id"), f"{path}.layer_id")
    return GradientUpdate(layer, gradients, trace)


def read_eligibility(obj: dict, path: str) -> list[float] | None:
    if "eligibility_trace" not in obj:
        return None
    if obj["eligibility_trace"] is None:
        return None
    trace = read_f32_list(obj["eligibility_trace"], f"{path}.eligibility_trace", MAX_CHANNEL_VALUES)
    check_finite_list(trace, f"{path}.eligibility_trace")
    return trace


def decode_traces(value: object) -> TraceBatch:
    obj = require_object(value, "EligibilityTraces")
    traces = read_traces(require_key(obj, "traces", "traces"))
    return TraceBatch(
        required_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        traces,
    )


def read_traces(value: object) -> list[TraceData]:
    if not isinstance(value, list):
        raise payload_error("traces must be an array")
    check_count(len(value), MAX_TRACES, "traces")
    traces = [decode_trace(item, index) for index, item in enumerate(value)]
    _unique_channels(traces)
    add_total(0, len(traces), "traces")
    return traces


def _unique_channels(traces: list[TraceData]) -> None:
    seen: set[int] = set()
    for trace in traces:
        if trace.channel_id in seen:
            raise payload_error(f"duplicate channel_id {trace.channel_id}")
        seen.add(trace.channel_id)


def decode_trace(value: object, index: int) -> TraceData:
    path = f"traces[{index}]"
    obj = require_object(value, path)
    data = TraceData(
        read_channel(obj, path),
        as_f32(require_key(obj, "trace_value", f"{path}.trace_value"), f"{path}.trace_value"),
        as_u32(require_key(obj, "last_spike_time", f"{path}.last_spike_time"), f"{path}.last_spike_time"),
    )
    check_finite(data.trace_value, f"{path}.trace_value")
    return data


def read_channel(obj: dict, path: str) -> int:
    if _both_channel_keys(obj):
        raise payload_error(f"{path} has both channel_id and neuron_id")
    if "channel_id" in obj:
        return as_u16(obj["channel_id"], f"{path}.channel_id")
    if "neuron_id" in obj:
        return as_u16(obj["neuron_id"], f"{path}.neuron_id")
    raise payload_error(f"missing {path}.channel_id")


def _both_channel_keys(obj: dict) -> bool:
    if "channel_id" not in obj:
        return False
    return "neuron_id" in obj
