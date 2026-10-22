"""Decoders for spike, embedding, stimulus, and neuromodulator payloads."""

from __future__ import annotations

from wire_v1.decode_meta import read_optional_metadata, validate_metadata
from wire_v1.errors import payload_error
from wire_v1.limits import MAX_AGGREGATE, MAX_CHANNEL_VALUES, MAX_SPIKE_EVENTS
from wire_v1.model import EmbeddingBatch, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch
from wire_v1.numbers import as_f32
from wire_v1.scalars import (
    add_total,
    as_i64,
    as_u16,
    as_u32,
    as_u64,
    check_count,
    check_finite,
    check_range,
    opt_session,
    require_key,
    require_object,
)


def decode_spikes(value: object) -> SpikeBatch:
    obj = require_object(value, "Spikes")
    spikes = read_spike_list(obj)
    batch = SpikeBatch(
        read_default_session(obj),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        as_u64(require_key(obj, "timestamp", "timestamp"), "timestamp"),
        spikes,
        read_optional_metadata(obj),
    )
    validate_metadata(batch.metadata, len(batch.spikes), "spikes")
    return batch


def read_default_session(obj: dict) -> str | None:
    if "session_id" not in obj:
        return None
    return opt_session(obj["session_id"], "session_id")


def read_spike_list(obj: dict) -> list[SpikeEvent]:
    raw = require_key(obj, "spikes", "spikes")
    if not isinstance(raw, list):
        raise payload_error("spikes must be an array")
    check_count(len(raw), MAX_SPIKE_EVENTS, "spikes")
    return [decode_spike(item, index) for index, item in enumerate(raw)]


def decode_spike(value: object, index: int) -> SpikeEvent:
    path = f"spikes[{index}]"
    obj = require_object(value, path)
    event = SpikeEvent(
        as_u16(require_key(obj, "channel", f"{path}.channel"), f"{path}.channel"),
        as_u32(require_key(obj, "time", f"{path}.time"), f"{path}.time"),
        as_f32(require_key(obj, "strength", f"{path}.strength"), f"{path}.strength"),
    )
    check_finite(event.strength, f"{path}.strength")
    return event


def decode_embeddings(value: object) -> EmbeddingBatch:
    obj = require_object(value, "Embeddings")
    embedding = read_f32_list(require_key(obj, "embedding", "embedding"), "embedding", MAX_CHANNEL_VALUES)
    sequence = as_u64(require_key(obj, "sequence_length", "sequence_length"), "sequence_length")
    check_count(sequence, MAX_AGGREGATE, "sequence_length")
    check_finite_list(embedding, "embedding")
    add_total(0, len(embedding), "embedding")
    return EmbeddingBatch(
        opt_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        embedding,
        sequence,
    )


def decode_stimuli(value: object) -> StimulusBatch:
    obj = require_object(value, "Stimuli")
    values = read_f32_list(require_key(obj, "values", "values"), "values", MAX_CHANNEL_VALUES)
    check_finite_list(values, "values")
    mask = read_valid_mask(require_key(obj, "valid_mask", "valid_mask"), values)
    metadata = read_optional_metadata(obj)
    validate_metadata(metadata, len(values), "values")
    return StimulusBatch(
        opt_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        as_u64(require_key(obj, "timestamp", "timestamp"), "timestamp"),
        values,
        mask,
        metadata,
    )


def read_valid_mask(value: object, values: list[float]) -> list[bool] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise payload_error("valid_mask must be an array or null")
    mask = _bool_list(value)
    if len(mask) != len(values):
        raise payload_error(
            f"valid_mask length {len(mask)} does not match values length {len(values)}"
        )
    return mask


def _bool_list(value: list) -> list[bool]:
    check_count(len(value), MAX_CHANNEL_VALUES, "valid_mask")
    mask: list[bool] = []
    for index, item in enumerate(value):
        mask.append(_one_bool(item, index))
    return mask


def _one_bool(item: object, index: int) -> bool:
    if not isinstance(item, bool):
        raise payload_error(f"valid_mask[{index}] must be a boolean")
    return item


def decode_neuromodulators(value: object) -> NeuromodulatorSnapshot:
    obj = require_object(value, "Neuromodulators")
    snapshot = NeuromodulatorSnapshot(
        as_i64(require_key(obj, "tick", "tick"), "tick"),
        as_f32(require_key(obj, "dopamine", "dopamine"), "dopamine"),
        as_f32(require_key(obj, "cortisol", "cortisol"), "cortisol"),
        as_f32(require_key(obj, "acetylcholine", "acetylcholine"), "acetylcholine"),
        as_f32(require_key(obj, "tempo", "tempo"), "tempo"),
    )
    check_neuromod_ranges(snapshot)
    return snapshot


def check_neuromod_ranges(snapshot: NeuromodulatorSnapshot) -> None:
    check_range(snapshot.dopamine, 0.0, 1.0, "dopamine")
    check_range(snapshot.cortisol, 0.0, 1.0, "cortisol")
    check_range(snapshot.acetylcholine, 0.0, 1.0, "acetylcholine")
    check_range(snapshot.tempo, 0.5, 2.0, "tempo")


def read_f32_list(value: object, path: str, limit: int) -> list[float]:
    if not isinstance(value, list):
        raise payload_error(f"{path} must be an array")
    check_count(len(value), limit, path)
    return [as_f32(item, f"{path}[{index}]") for index, item in enumerate(value)]


def check_finite_list(values: list[float], path: str) -> None:
    for index, item in enumerate(values):
        check_finite(item, f"{path}[{index}]")
