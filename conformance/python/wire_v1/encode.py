"""
Canonical re-encode of a decoded wire-v1 message.
"""

from __future__ import annotations

from wire_v1.errors import payload_error
from wire_v1.jsonio import dumps
from wire_v1.limits import CURRENT, UNIT_VARIANTS
from wire_v1.model import (
    ConfigPayload,
    ConfigValue,
    EmbeddingBatch,
    GradientBatch,
    Message,
    NeuromodulatorSnapshot,
    SpikeBatch,
    StimulusBatch,
    TraceBatch,
)


def encode_canonical(message: Message) -> bytes:
    text = dumps({"payload": encode_payload(message), "wire_version": CURRENT})
    return text.encode("utf-8")


def encode_payload(message: Message) -> object:
    if message.variant in UNIT_VARIANTS:
        return message.variant
    encoder = ENCODERS.get(message.variant)
    if encoder is None:
        raise payload_error(f"cannot encode variant {message.variant}")
    return encoder(message.body)


def encode_loss(body: object) -> dict:
    return {"Loss": body}


def encode_spikes(body: object) -> dict:
    if not isinstance(body, SpikeBatch):
        raise payload_error("cannot encode Spikes")
    return {"Spikes": _spike_body(body)}


def _spike_body(body: SpikeBatch) -> dict:
    return {
        "session_id": body.session_id,
        "batch_id": body.batch_id,
        "timestamp": body.timestamp,
        "spikes": [_spike_event(spike) for spike in body.spikes],
        "metadata": _encode_metadata(body.metadata),
    }


def _spike_event(spike) -> dict:
    return {"channel": spike.channel, "time": spike.time, "strength": spike.strength}


def encode_embeddings(body: object) -> dict:
    if not isinstance(body, EmbeddingBatch):
        raise payload_error("cannot encode Embeddings")
    return {
        "Embeddings": {
            "session_id": body.session_id,
            "batch_id": body.batch_id,
            "embedding": body.embedding,
            "sequence_length": body.sequence_length,
        }
    }


def encode_stimuli(body: object) -> dict:
    if not isinstance(body, StimulusBatch):
        raise payload_error("cannot encode Stimuli")
    return {"Stimuli": _stimulus_body(body)}


def _stimulus_body(body: StimulusBatch) -> dict:
    return {
        "session_id": body.session_id,
        "batch_id": body.batch_id,
        "timestamp": body.timestamp,
        "values": body.values,
        "valid_mask": body.valid_mask,
        "metadata": _encode_metadata(body.metadata),
    }


def encode_neuromodulators(body: object) -> dict:
    if not isinstance(body, NeuromodulatorSnapshot):
        raise payload_error("cannot encode Neuromodulators")
    return {
        "Neuromodulators": {
            "tick": body.tick,
            "dopamine": body.dopamine,
            "cortisol": body.cortisol,
            "acetylcholine": body.acetylcholine,
            "tempo": body.tempo,
        }
    }


def encode_config(body: object) -> dict:
    if not isinstance(body, ConfigPayload):
        raise payload_error("cannot encode ConfigUpdate")
    return {
        "ConfigUpdate": {
            "session_id": body.session_id,
            "config": {key: _config_value(item) for key, item in body.config.items()},
        }
    }


def _config_value(value: ConfigValue) -> object:
    if value.kind in ("float", "array", "bool", "string"):
        return value.value
    raise payload_error(f"cannot encode config value {value.kind}")


def encode_gradients(body: object) -> dict:
    if not isinstance(body, GradientBatch):
        raise payload_error("cannot encode GradientUpdate")
    return {
        "GradientUpdate": {
            "session_id": body.session_id,
            "batch_id": body.batch_id,
            "gradients": [_gradient_row(row) for row in body.gradients],
        }
    }


def _gradient_row(row) -> dict:
    return {
        "layer_id": row.layer_id,
        "gradients": row.gradients,
        "eligibility_trace": row.eligibility_trace,
    }


def encode_traces(body: object) -> dict:
    if not isinstance(body, TraceBatch):
        raise payload_error("cannot encode EligibilityTraces")
    return {
        "EligibilityTraces": {
            "session_id": body.session_id,
            "batch_id": body.batch_id,
            "traces": [_trace_row(trace) for trace in body.traces],
        }
    }


def _trace_row(trace) -> dict:
    return {
        "channel_id": trace.channel_id,
        "trace_value": trace.trace_value,
        "last_spike_time": trace.last_spike_time,
    }


def _encode_metadata(metadata) -> object:
    if metadata is None:
        return None
    return {
        "processing_latency_ns": metadata.processing_latency_ns,
        "source": metadata.source,
        "custom": metadata.custom,
    }


ENCODERS = {
    "Loss": encode_loss,
    "Spikes": encode_spikes,
    "Embeddings": encode_embeddings,
    "Stimuli": encode_stimuli,
    "Neuromodulators": encode_neuromodulators,
    "ConfigUpdate": encode_config,
    "GradientUpdate": encode_gradients,
    "EligibilityTraces": encode_traces,
}
