"""
Decoded wire-v1 message shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


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
