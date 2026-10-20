"""Canonical re-encode of a decoded wire-v1 message."""

from __future__ import annotations

from dataclasses import asdict

from wire_v1.errors import payload_error
from wire_v1.jsonio import dumps
from wire_v1.limits import CURRENT, UNIT_VARIANTS
from wire_v1.model import (
    ConfigPayload,
    EmbeddingBatch,
    GradientBatch,
    Message,
    NeuromodulatorSnapshot,
    SpikeBatch,
    StimulusBatch,
    TraceBatch,
)

_STRUCTS = {
    "Spikes": SpikeBatch,
    "Embeddings": EmbeddingBatch,
    "Stimuli": StimulusBatch,
    "Neuromodulators": NeuromodulatorSnapshot,
    "GradientUpdate": GradientBatch,
    "EligibilityTraces": TraceBatch,
}


def encode_canonical(message: Message) -> bytes:
    text = dumps({"payload": encode_payload(message), "wire_version": CURRENT})
    return text.encode("utf-8")


def encode_payload(message: Message) -> object:
    if message.variant in UNIT_VARIANTS:
        return message.variant
    if message.variant == "Loss":
        return {"Loss": message.body}
    if message.variant == "ConfigUpdate":
        return _encode_config(message.body)
    return _encode_struct(message)


def _encode_struct(message: Message) -> dict:
    expected = _STRUCTS.get(message.variant)
    if expected is None:
        raise payload_error(f"cannot encode variant {message.variant}")
    if not isinstance(message.body, expected):
        raise payload_error(f"cannot encode {message.variant}")
    return {message.variant: asdict(message.body)}


def _encode_config(body: object) -> dict:
    if not isinstance(body, ConfigPayload):
        raise payload_error("cannot encode ConfigUpdate")
    values = {key: item.value for key, item in body.config.items()}
    return {"ConfigUpdate": {"session_id": body.session_id, "config": values}}
