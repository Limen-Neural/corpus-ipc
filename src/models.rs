// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data types that flow over the compute backend IPC wire.
//!
//! # Transport types vs training types
//!
//! [`SpikeBatch`] and [`TraceBatch`] in this crate are **IPC transport / wire
//! payloads**. They are owned by `corpus-ipc` and serialized on the hybrid-flow
//! message bus (`IpcMessage::Spikes`, `IpcMessage::EligibilityTraces`).
//!
//! They are **not** the same types as `SpikeBatch` / `TraceBatch` in
//! [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)
//! (`src/types.jl`). Those are **training-facing** in-memory batches:
//!
//! | | `corpus-ipc` (this crate) | `SynapticDistill.jl` |
//! |---|---------------------------|----------------------|
//! | Domain | IPC wire / session routing | SNN training step |
//! | `SpikeBatch` | `session_id`, `batch_id`, timestamped [`SpikeEvent`]s | spike trains (`spikes`), optional `times` / `targets` |
//! | `TraceBatch` | `session_id`, `batch_id`, typed [`TraceData`] rows | unstructured `traces` for e-prop / credit assignment |
//!
//! Keep the Rust names `SpikeBatch` / `TraceBatch` so serde identifiers and
//! existing imports stay stable. Prefer [`IpcSpikeBatch`] / [`IpcTraceBatch`]
//! when the cross-repo collision would otherwise be unclear. Do not merge the
//! IPC and training definitions.

use serde::{Deserialize, Deserializer, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, bounded_map, bounded_opt_string,
    bounded_opt_vec, bounded_string, bounded_vec, check_count, check_finite, check_finite_slice,
    check_opt_string, check_range, check_string, check_unique_by, check_unique_strings,
    finite_f32_at,
};

fn de_opt_session_id<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    bounded_opt_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "session_id")
}

fn de_session_id<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "session_id")
}

fn de_source<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    bounded_opt_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "source")
}

fn de_layer_id<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "layer_id")
}

fn de_values<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "values")
}

fn de_embedding<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "embedding")
}

fn de_valid_mask<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<bool>>, D::Error> {
    bounded_opt_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "valid_mask")
}

fn de_spikes<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<SpikeEvent>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_spike_events, "spikes")
}

fn de_traces<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<TraceData>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_traces, "traces")
}

fn de_gradient_rows<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<GradientUpdate>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_gradients, "gradients")
}

fn de_gradient_values<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "gradients")
}

fn de_eligibility_trace<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f32>>, D::Error> {
    bounded_opt_vec(
        d,
        ProtocolLimits::DEFAULT.max_channel_values,
        "eligibility_trace",
    )
}

fn de_float_array<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "config")
}

fn de_config_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "config")
}

fn de_metadata_custom<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<std::collections::HashMap<String, String>, D::Error> {
    bounded_map(
        d,
        ProtocolLimits::DEFAULT.max_metadata_entries,
        ProtocolLimits::DEFAULT.max_string_bytes,
        "custom",
    )
}

fn de_config_map<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<std::collections::HashMap<String, ConfigValue>, D::Error> {
    bounded_map(
        d,
        ProtocolLimits::DEFAULT.max_metadata_entries,
        ProtocolLimits::DEFAULT.max_string_bytes,
        "config",
    )
}

fn de_finite_f32<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "value")
}

fn de_loss<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "Loss")
}

fn de_strength<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "strength")
}

fn de_trace_value<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "trace_value")
}

/// 4-runtime snapshot decoded from the remote compute's 88-byte generic packet.
///
/// # Wire format (bytes 72–87 of the generic IPC packet)
/// ```text
/// [72..76]  dopamine       f32 LE   reward / learning-rate gate
/// [76..80]  cortisol       f32 LE   stress / inhibition
/// [80..84]  acetylcholine  f32 LE   focus / signal-to-noise
/// [84..88]  tempo          f32 LE   clock-driven timing scale
/// ```
///
/// # References
///
/// - Schultz, W. (1998). Predictive reward signal of dopamine channels.
///   *Journal of Neurophysiology*, 80(1), 1–27.
/// - Arnsten, A. F. T. (2009). Stress signalling pathways that impair
///   prefrontal cortex structure and function.
///   *Nature Reviews Neuroscience*, 10(6), 410–422.
/// - Hasselmo, M. E. (1999). Neuromodulation: acetylcholine and memory
///   consolidation. *Trends in Cognitive Sciences*, 3(9), 351–359.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NeuromodulatorSnapshotWire")]
pub struct NeuromodulatorSnapshot {
    /// Tick counter from the remote compute (monotonically increasing).
    pub tick: i64,
    /// Dopamine level (reward / STDP learning-rate gate). Range [0, 1].
    pub dopamine: f32,
    /// Cortisol level (thermal/power stress inhibition). Range [0, 1].
    pub cortisol: f32,
    /// Acetylcholine level (focus / signal-to-noise ratio). Range [0, 1].
    pub acetylcholine: f32,
    /// Tempo scale (clock-driven timing; 1.0 = nominal). Range [0.5, 2.0].
    pub tempo: f32,
}

/// Deserialization-only shadow of [`NeuromodulatorSnapshot`] with the
/// identical wire shape. `NeuromodulatorSnapshot`'s real `Deserialize` impl
/// goes through this type and [`NeuromodulatorSnapshot::validate`] so an
/// out-of-range or non-finite field is rejected at deserialization instead
/// of silently reaching consumers (this type is reachable via
/// [`IpcMessage::Neuromodulators`]).
#[derive(Deserialize)]
struct NeuromodulatorSnapshotWire {
    tick: i64,
    dopamine: f32,
    cortisol: f32,
    acetylcholine: f32,
    tempo: f32,
}

impl TryFrom<NeuromodulatorSnapshotWire> for NeuromodulatorSnapshot {
    type Error = ValidationError;

    fn try_from(wire: NeuromodulatorSnapshotWire) -> Result<Self, Self::Error> {
        let snapshot = NeuromodulatorSnapshot {
            tick: wire.tick,
            dopamine: wire.dopamine,
            cortisol: wire.cortisol,
            acetylcholine: wire.acetylcholine,
            tempo: wire.tempo,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

impl NeuromodulatorSnapshot {
    /// Parse from the 4 generic score floats in bytes `[72..88]` of a generic packet.
    ///
    /// Calls [`Validate::validate`] internally and returns `Err` if the decoded
    /// bytes are out of the documented ranges or non-finite. This keeps
    /// byte-packet ingress consistent with JSON ingress (`IpcMessage::Neuromodulators`
    /// deserialization also validates) — otherwise a bad packet would only
    /// surface as a confusing "failed to deserialize" error at the receiver,
    /// pointing away from where the bad bytes actually came from.
    pub fn from_scores(tick: i64, scores: &[f32; 4]) -> Result<Self, ValidationError> {
        let snapshot = Self {
            tick,
            dopamine: scores[0],
            cortisol: scores[1],
            acetylcholine: scores[2],
            tempo: scores[3],
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

impl Validate for NeuromodulatorSnapshot {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_range("dopamine", self.dopamine, 0.0, 1.0)?;
        check_range("cortisol", self.cortisol, 0.0, 1.0)?;
        check_range("acetylcholine", self.acetylcholine, 0.0, 1.0)?;
        check_range("tempo", self.tempo, 0.5, 2.0)?;
        Ok(())
    }
}

/// Core message enum for cross-process IPC.
///
/// Messages are separated into:
/// - Input messages (spikes, embeddings, stimuli, neuromodulators, config)
/// - Output messages (gradients, traces, training status)
/// - Control messages (shutdown, ping)
///
/// Variant names (`Spikes`, `EligibilityTraces`, …) are serde identifiers and
/// must stay stable. The payloads they wrap are IPC transport types, not
/// SynapticDistill training structs (see the module docs).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum IpcMessage {
    // Input messages
    /// Wire envelope for an IPC [`SpikeBatch`] (not a SynapticDistill training batch).
    Spikes(SpikeBatch),
    Embeddings(EmbeddingBatch),
    /// Typed continuous runtime stimulus ingress (e.g. `thalamic-relay` ->
    /// `brainstem-daemon`). See [`StimulusBatch`] for channel-width and
    /// invalid/missing-channel semantics.
    Stimuli(StimulusBatch),
    /// Typed neuromodulator ingress, replacing an unstructured float tail.
    /// See [`NeuromodulatorSnapshot`] and [`Validate`].
    Neuromodulators(NeuromodulatorSnapshot),
    Loss(#[serde(deserialize_with = "de_loss")] f32),
    ConfigUpdate(ConfigPayload),

    // Output messages
    GradientUpdate(GradientBatch),
    /// Wire envelope for an IPC [`TraceBatch`] (not a SynapticDistill training batch).
    EligibilityTraces(TraceBatch),
    TrainingComplete,

    // Control
    Shutdown,
    Ping,
}

/// Canonical wire batch of continuous, domain-neutral runtime stimulus values.
///
/// This is the typed replacement for downstream services' ad-hoc stimulus
/// payloads — e.g. `thalamic-relay`'s untagged `{"type":"Stimuli","values":[...]}`
/// UDP JSON, and `brainstem-daemon`'s local `IngressPacket { stimuli, modulators }`
/// struct. `corpus-ipc` owns this schema; downstream services should decode/encode
/// through [`IpcMessage::Stimuli`] instead of a private struct or raw
/// `serde_json::Value` field indexing.
///
/// # Channel width
///
/// `values.len()` is the channel count. It is **not fixed** by this crate —
/// do not encode any particular network's input width (e.g. Spikenaut's
/// current axon count) into this type. Consumers determine width at runtime
/// from the batch itself.
///
/// # Invalid / missing channel semantics
///
/// `valid_mask`, when `Some`, must be the same length as `values`.
/// `valid_mask[i] == false` means channel `i` has no valid reading this tick;
/// the corresponding `values[i]` is a placeholder (`0.0` by convention) and
/// must **not** be interpreted as a real zero-valued reading. When
/// `valid_mask` is `None`, every entry in `values` is valid.
///
/// This is a deliberate improvement over ad-hoc formats (e.g. `thalamic-relay`'s
/// UDP handler) that silently coerce missing or non-numeric channels to `0.0`
/// with no way to distinguish "sensor read zero" from "no data this tick."
///
/// Fields are public (matching this crate's other wire batches, e.g.
/// [`SpikeBatch`]) for direct Rust construction, but **deserialization
/// enforces the `valid_mask`-length invariant**: a JSON/wire payload with a
/// `valid_mask` whose length differs from `values.len()` fails to
/// deserialize (via [`StimulusBatch::validate`] through a `TryFrom` shadow
/// type), rather than silently producing an inconsistent instance. Call
/// [`Validate::validate`] explicitly after constructing one directly in
/// Rust (which bypasses deserialization) to get the same check.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "StimulusBatchWire")]
pub struct StimulusBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Timestamp in nanoseconds (UTC or relative).
    pub timestamp: u64,
    /// Continuous stimulus values. Length is the channel count; not fixed by this crate.
    pub values: Vec<f32>,
    /// Optional per-channel validity mask, same length as `values` when present.
    /// `false` marks a channel as invalid/missing for this tick (see type docs).
    pub valid_mask: Option<Vec<bool>>,
    /// Optional batch-level metadata.
    pub metadata: Option<BatchMetadata>,
}

impl Validate for StimulusBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("values", self.values.len(), limits.max_channel_values)?;
        check_finite_slice("values", &self.values)?;
        if let Some(mask) = &self.valid_mask {
            check_count("valid_mask", mask.len(), limits.max_channel_values)?;
            if mask.len() != self.values.len() {
                return Err(ValidationError::length_mismatch(
                    "valid_mask",
                    mask.len(),
                    self.values.len(),
                ));
            }
        }
        validate_optional_metadata(&self.metadata, limits, self.values.len())
    }
}

fn validate_optional_metadata(
    metadata: &Option<BatchMetadata>,
    limits: ProtocolLimits,
    already_counted: usize,
) -> Result<(), ValidationError> {
    let mut total = 0;
    add_to_total(&mut total, already_counted, limits, "aggregate")?;
    if let Some(metadata) = metadata {
        metadata.validate_with(limits)?;
        add_to_total(&mut total, metadata.custom.len(), limits, "aggregate")?;
    }
    Ok(())
}

/// Deserialization-only shadow of [`StimulusBatch`] with the identical wire
/// shape. `StimulusBatch`'s real `Deserialize` impl goes through this type and
/// [`Validate::validate`] so a malformed `valid_mask` length, non-finite
/// value, or oversize payload is rejected at deserialization.
#[derive(Deserialize)]
struct StimulusBatchWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    timestamp: u64,
    #[serde(deserialize_with = "de_values")]
    values: Vec<f32>,
    #[serde(deserialize_with = "de_valid_mask")]
    valid_mask: Option<Vec<bool>>,
    metadata: Option<BatchMetadata>,
}

impl TryFrom<StimulusBatchWire> for StimulusBatch {
    type Error = ValidationError;

    fn try_from(wire: StimulusBatchWire) -> Result<Self, Self::Error> {
        let batch = StimulusBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            timestamp: wire.timestamp,
            values: wire.values,
            valid_mask: wire.valid_mask,
            metadata: wire.metadata,
        };
        batch.validate()?;
        Ok(batch)
    }
}

/// IPC transport batch of spike events from compute processing.
///
/// This is a **wire-level** payload for [`IpcMessage::Spikes`]. Field names
/// and layout are part of the serialized protocol and must not be changed
/// casually.
///
/// **Not** [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)'s
/// training `SpikeBatch` (`spikes` / `times` / `targets`). That type is an
/// in-memory training batch; this type is a session-correlated list of
/// [`SpikeEvent`]s. Use [`IpcSpikeBatch`] when the name collision matters.
///
/// Names stay `SpikeBatch` so existing Rust imports and serde identifiers
/// remain compatible (RM-324 / #7).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "SpikeBatchWire")]
pub struct SpikeBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Timestamp in nanoseconds (UTC or relative).
    pub timestamp: u64,
    /// Individual spike events.
    pub spikes: Vec<SpikeEvent>,
    /// Optional batch-level metadata.
    pub metadata: Option<BatchMetadata>,
}

#[derive(Deserialize)]
struct SpikeBatchWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    timestamp: u64,
    #[serde(deserialize_with = "de_spikes")]
    spikes: Vec<SpikeEvent>,
    metadata: Option<BatchMetadata>,
}

impl TryFrom<SpikeBatchWire> for SpikeBatch {
    type Error = ValidationError;

    fn try_from(wire: SpikeBatchWire) -> Result<Self, Self::Error> {
        let batch = SpikeBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            timestamp: wire.timestamp,
            spikes: wire.spikes,
            metadata: wire.metadata,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for SpikeBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("spikes", self.spikes.len(), limits.max_spike_events)?;
        for (index, spike) in self.spikes.iter().enumerate() {
            spike
                .validate_with(limits)
                .map_err(|err| prefix_path(err, &format!("spikes[{index}]")))?;
        }
        validate_optional_metadata(&self.metadata, limits, self.spikes.len())
    }
}

/// Explicit IPC-domain name for [`SpikeBatch`].
///
/// Same type and same wire format. Prefer this alias in new code that sits
/// next to SynapticDistill training types.
pub type IpcSpikeBatch = SpikeBatch;

/// Individual spike event with channel, timing, and strength.
///
/// An element of an IPC [`SpikeBatch`], not a SynapticDistill training row.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "SpikeEventWire")]
pub struct SpikeEvent {
    /// Compute channel or channel identifier.
    pub channel: u16,
    /// Spike timestamp (relative or absolute).
    pub time: u32,
    /// Spike strength or amplitude.
    pub strength: f32,
}

#[derive(Deserialize)]
struct SpikeEventWire {
    channel: u16,
    time: u32,
    #[serde(deserialize_with = "de_strength")]
    strength: f32,
}

impl TryFrom<SpikeEventWire> for SpikeEvent {
    type Error = ValidationError;

    fn try_from(wire: SpikeEventWire) -> Result<Self, Self::Error> {
        let event = SpikeEvent {
            channel: wire.channel,
            time: wire.time,
            strength: wire.strength,
        };
        event.validate()?;
        Ok(event)
    }
}

impl Validate for SpikeEvent {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_finite("strength", self.strength)
    }
}

/// Batch of embeddings for projector and transformer components.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "EmbeddingBatchWire")]
pub struct EmbeddingBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Embedding vector from compute processing.
    pub embedding: Vec<f32>,
    /// Sequence length for transformer compatibility.
    pub sequence_length: usize,
}

#[derive(Deserialize)]
struct EmbeddingBatchWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    #[serde(deserialize_with = "de_embedding")]
    embedding: Vec<f32>,
    sequence_length: usize,
}

impl TryFrom<EmbeddingBatchWire> for EmbeddingBatch {
    type Error = ValidationError;

    fn try_from(wire: EmbeddingBatchWire) -> Result<Self, Self::Error> {
        let batch = EmbeddingBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            embedding: wire.embedding,
            sequence_length: wire.sequence_length,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for EmbeddingBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("embedding", self.embedding.len(), limits.max_channel_values)?;
        check_count(
            "sequence_length",
            self.sequence_length,
            limits.max_aggregate_records,
        )?;
        check_finite_slice("embedding", &self.embedding)?;
        let mut total = 0;
        add_to_total(&mut total, self.embedding.len(), limits, "aggregate")?;
        Ok(())
    }
}

/// Gradient update batch from external training or optimization algorithms.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "GradientBatchWire")]
pub struct GradientBatch {
    /// Session ID for routing back to correct experiment.
    pub session_id: String,
    /// Batch ID correlation with original input.
    pub batch_id: u64,
    /// Individual gradient updates.
    pub gradients: Vec<GradientUpdate>,
}

#[derive(Deserialize)]
struct GradientBatchWire {
    #[serde(deserialize_with = "de_session_id")]
    session_id: String,
    batch_id: u64,
    #[serde(deserialize_with = "de_gradient_rows")]
    gradients: Vec<GradientUpdate>,
}

impl TryFrom<GradientBatchWire> for GradientBatch {
    type Error = ValidationError;

    fn try_from(wire: GradientBatchWire) -> Result<Self, Self::Error> {
        let batch = GradientBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            gradients: wire.gradients,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for GradientBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("session_id", &self.session_id, limits)?;
        check_count("gradients", self.gradients.len(), limits.max_gradients)?;
        let mut total = 0;
        add_to_total(&mut total, self.gradients.len(), limits, "aggregate")?;
        accumulate_gradient_rows(&self.gradients, limits, &mut total)?;
        check_unique_strings("gradients", &self.gradients, |row| row.layer_id.as_str())?;
        Ok(())
    }
}

fn accumulate_gradient_rows(
    rows: &[GradientUpdate],
    limits: ProtocolLimits,
    total: &mut usize,
) -> Result<(), ValidationError> {
    for (index, update) in rows.iter().enumerate() {
        update
            .validate_with(limits)
            .map_err(|err| prefix_path(err, &format!("gradients[{index}]")))?;
        add_to_total(total, update.gradients.len(), limits, "aggregate")?;
        if let Some(trace) = &update.eligibility_trace {
            add_to_total(total, trace.len(), limits, "aggregate")?;
        }
    }
    Ok(())
}

/// Individual gradient update for a specific layer or parameter.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "GradientUpdateWire")]
pub struct GradientUpdate {
    /// Target layer identifier.
    pub layer_id: String,
    /// Gradient values (flattened or sparse representation).
    pub gradients: Vec<f32>,
    /// Optional eligibility trace for E-prop algorithms.
    pub eligibility_trace: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct GradientUpdateWire {
    #[serde(deserialize_with = "de_layer_id")]
    layer_id: String,
    #[serde(deserialize_with = "de_gradient_values")]
    gradients: Vec<f32>,
    #[serde(default, deserialize_with = "de_eligibility_trace")]
    eligibility_trace: Option<Vec<f32>>,
}

impl TryFrom<GradientUpdateWire> for GradientUpdate {
    type Error = ValidationError;

    fn try_from(wire: GradientUpdateWire) -> Result<Self, Self::Error> {
        let update = GradientUpdate {
            layer_id: wire.layer_id,
            gradients: wire.gradients,
            eligibility_trace: wire.eligibility_trace,
        };
        update.validate()?;
        Ok(update)
    }
}

impl Validate for GradientUpdate {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("layer_id", &self.layer_id, limits)?;
        check_count("gradients", self.gradients.len(), limits.max_channel_values)?;
        check_finite_slice("gradients", &self.gradients)?;
        if let Some(trace) = &self.eligibility_trace {
            check_count("eligibility_trace", trace.len(), limits.max_channel_values)?;
            check_finite_slice("eligibility_trace", trace)?;
        }
        Ok(())
    }
}

/// IPC transport batch of eligibility traces for credit assignment.
///
/// This is a **wire-level** payload for [`IpcMessage::EligibilityTraces`].
/// Field names and layout are part of the serialized protocol.
///
/// **Not** [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)'s
/// training `TraceBatch` (a single unstructured `traces` field). This type
/// carries `session_id` / `batch_id` plus typed [`TraceData`] rows. Use
/// [`IpcTraceBatch`] when the name collision matters.
///
/// Names stay `TraceBatch` so existing Rust imports and serde identifiers
/// remain compatible (RM-324 / #7).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "TraceBatchWire")]
pub struct TraceBatch {
    /// Session ID for routing.
    pub session_id: String,
    /// Batch ID correlation.
    pub batch_id: u64,
    /// Eligibility trace data.
    pub traces: Vec<TraceData>,
}

/// Explicit IPC-domain name for [`TraceBatch`].
///
/// Same type and same wire format. Prefer this alias in new code that sits
/// next to SynapticDistill training types.
pub type IpcTraceBatch = TraceBatch;

#[derive(Deserialize)]
struct TraceBatchWire {
    #[serde(deserialize_with = "de_session_id")]
    session_id: String,
    batch_id: u64,
    #[serde(deserialize_with = "de_traces")]
    traces: Vec<TraceData>,
}

impl TryFrom<TraceBatchWire> for TraceBatch {
    type Error = ValidationError;

    fn try_from(wire: TraceBatchWire) -> Result<Self, Self::Error> {
        let batch = TraceBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            traces: wire.traces,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for TraceBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("session_id", &self.session_id, limits)?;
        check_count("traces", self.traces.len(), limits.max_traces)?;
        let mut total = 0;
        add_to_total(&mut total, self.traces.len(), limits, "aggregate")?;
        for (index, trace) in self.traces.iter().enumerate() {
            trace
                .validate_with(limits)
                .map_err(|err| prefix_path(err, &format!("traces[{index}]")))?;
        }
        check_unique_by("traces", &self.traces, |row| row.channel_id)?;
        Ok(())
    }
}

/// Individual eligibility trace data on the IPC wire.
///
/// An element of an IPC [`TraceBatch`], not a SynapticDistill training trace.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "TraceDataWire")]
pub struct TraceData {
    /// Channel or synapse identifier.
    #[serde(alias = "neuron_id")]
    pub channel_id: u16,
    /// Trace value (decay-modulated spike history).
    pub trace_value: f32,
    /// Timestamp of last contributing spike.
    pub last_spike_time: u32,
}

#[derive(Deserialize)]
struct TraceDataWire {
    #[serde(alias = "neuron_id")]
    channel_id: u16,
    #[serde(deserialize_with = "de_trace_value")]
    trace_value: f32,
    last_spike_time: u32,
}

impl TryFrom<TraceDataWire> for TraceData {
    type Error = ValidationError;

    fn try_from(wire: TraceDataWire) -> Result<Self, Self::Error> {
        let data = TraceData {
            channel_id: wire.channel_id,
            trace_value: wire.trace_value,
            last_spike_time: wire.last_spike_time,
        };
        data.validate()?;
        Ok(data)
    }
}

impl Validate for TraceData {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_finite("trace_value", self.trace_value)
    }
}

/// Configuration payload for runtime parameter updates.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "ConfigPayloadWire")]
pub struct ConfigPayload {
    /// Target session (None = global).
    pub session_id: Option<String>,
    /// Configuration key-value pairs.
    pub config: std::collections::HashMap<String, ConfigValue>,
}

#[derive(Deserialize)]
struct ConfigPayloadWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    #[serde(deserialize_with = "de_config_map")]
    config: std::collections::HashMap<String, ConfigValue>,
}

impl TryFrom<ConfigPayloadWire> for ConfigPayload {
    type Error = ValidationError;

    fn try_from(wire: ConfigPayloadWire) -> Result<Self, Self::Error> {
        let payload = ConfigPayload {
            session_id: wire.session_id,
            config: wire.config,
        };
        payload.validate()?;
        Ok(payload)
    }
}

impl Validate for ConfigPayload {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("config", self.config.len(), limits.max_metadata_entries)?;
        let mut total = 0;
        add_to_total(&mut total, self.config.len(), limits, "aggregate")?;
        validate_config_entries(&self.config, limits, &mut total)
    }
}

fn validate_config_entries(
    config: &std::collections::HashMap<String, ConfigValue>,
    limits: ProtocolLimits,
    total: &mut usize,
) -> Result<(), ValidationError> {
    for (key, value) in config {
        if key.is_empty() {
            return Err(ValidationError::nested_metadata("config.<empty>"));
        }
        check_string("config.<key>", key, limits)?;
        value
            .validate_with(limits)
            .map_err(|err| prefix_path(err, &format!("config.{key}")))?;
        if let ConfigValue::FloatArray(values) = value {
            add_to_total(total, values.len(), limits, "aggregate")?;
        }
    }
    Ok(())
}

/// Configuration value types.
///
/// Uses `#[serde(untagged)]` so plain JSON numbers/strings/arrays/booleans
/// work directly inside `ConfigPayload::config`.
///
/// **Untagged deserialization behavior (intentional, pre-existing):**
/// `Float(f32)` is first, so JSON numbers (e.g. `42` or `1.5`) always
/// deserialize as `Float`. `Integer` is only reached for values that were
/// originally `ConfigValue::Integer` in Rust and then serialized, or under
/// certain deserializer configurations.
///
/// Round-tripping `Integer(42)` through JSON yields `Float(42.0)`.
/// Large integers (> ~2^24) may lose precision in f32.
/// Consumers relying on exact integer identity should be aware.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ConfigValue {
    /// Floating-point value.
    ///
    /// Because this is the first variant in an untagged enum, JSON
    /// numbers (integers and floats) deserialize as `Float`.
    Float(#[serde(deserialize_with = "de_finite_f32")] f32),

    /// Integer value (u64).
    ///
    /// Typically only produced when a Rust `ConfigValue::Integer` is
    /// serialized and round-tripped with the same serde configuration,
    /// or in specific deserializer contexts. Plain JSON numbers land
    /// in `Float` due to declaration order.
    Integer(u64),

    /// String value.
    ///
    /// Allows string-valued config (e.g. mode names, paths) in `ConfigPayload::config`.
    String(#[serde(deserialize_with = "de_config_string")] String),

    /// Boolean value.
    Boolean(bool),
    FloatArray(#[serde(deserialize_with = "de_float_array")] Vec<f32>),
}

impl Validate for ConfigValue {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        match self {
            Self::Float(value) => check_finite("value", *value),
            Self::Integer(_) | Self::Boolean(_) => Ok(()),
            Self::String(value) => check_string("value", value, limits),
            Self::FloatArray(values) => {
                check_count("value", values.len(), limits.max_channel_values)?;
                check_finite_slice("value", values)
            }
        }
    }
}

/// Optional batch metadata for debugging and monitoring.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "BatchMetadataWire")]
pub struct BatchMetadata {
    /// Processing latency in nanoseconds.
    pub processing_latency_ns: Option<u64>,
    /// Source identifier (for example, "encoder", "compute_layer_2").
    pub source: Option<String>,
    /// Additional metadata fields.
    pub custom: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
struct BatchMetadataWire {
    processing_latency_ns: Option<u64>,
    #[serde(default, deserialize_with = "de_source")]
    source: Option<String>,
    #[serde(default, deserialize_with = "de_metadata_custom")]
    custom: std::collections::HashMap<String, String>,
}

impl TryFrom<BatchMetadataWire> for BatchMetadata {
    type Error = ValidationError;

    fn try_from(wire: BatchMetadataWire) -> Result<Self, Self::Error> {
        let metadata = BatchMetadata {
            processing_latency_ns: wire.processing_latency_ns,
            source: wire.source,
            custom: wire.custom,
        };
        metadata.validate()?;
        Ok(metadata)
    }
}

impl Validate for BatchMetadata {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("source", self.source.as_deref(), limits)?;
        check_count("custom", self.custom.len(), limits.max_metadata_entries)?;
        for (key, value) in &self.custom {
            if key.is_empty() {
                return Err(ValidationError::nested_metadata("custom.<empty>"));
            }
            check_string("custom.<key>", key, limits)?;
            check_string("custom.<value>", value, limits)?;
        }
        Ok(())
    }
}

impl Validate for IpcMessage {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        match self {
            Self::Spikes(batch) => batch.validate_with(limits),
            Self::Embeddings(batch) => batch.validate_with(limits),
            Self::Stimuli(batch) => batch.validate_with(limits),
            Self::Neuromodulators(snapshot) => snapshot.validate_with(limits),
            Self::ConfigUpdate(payload) => payload.validate_with(limits),
            Self::GradientUpdate(batch) => batch.validate_with(limits),
            Self::EligibilityTraces(batch) => batch.validate_with(limits),
            Self::Loss(value) => check_finite("Loss", *value),
            Self::TrainingComplete | Self::Shutdown | Self::Ping => Ok(()),
        }
    }
}

fn prefix_path(mut err: ValidationError, prefix: &str) -> ValidationError {
    if err.path.is_empty() {
        err.path = prefix.to_owned();
    } else {
        err.path = format!("{prefix}.{}", err.path);
    }
    err
}

#[cfg(test)]
#[path = "models_limits_tests.rs"]
mod limits_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spike_batch() -> SpikeBatch {
        SpikeBatch {
            session_id: Some("sess-1".into()),
            batch_id: 7,
            timestamp: 1_700_000_000,
            spikes: vec![SpikeEvent {
                channel: 3,
                time: 11,
                strength: 0.5,
            }],
            metadata: None,
        }
    }

    fn sample_stimulus_batch() -> StimulusBatch {
        StimulusBatch {
            session_id: Some("sess-1".into()),
            batch_id: 7,
            timestamp: 1_700_000_000,
            values: vec![0.5, 0.0, -0.25],
            valid_mask: Some(vec![true, false, true]),
            metadata: None,
        }
    }

    fn sample_neuromodulator_snapshot() -> NeuromodulatorSnapshot {
        NeuromodulatorSnapshot {
            tick: 42,
            dopamine: 0.4,
            cortisol: 0.3,
            acetylcholine: 0.2,
            tempo: 1.0,
        }
    }

    #[test]
    fn neuromodulator_snapshot_validate_accepts_in_range_values() {
        assert!(sample_neuromodulator_snapshot().validate().is_ok());
    }

    #[test]
    fn neuromodulator_snapshot_from_scores_accepts_in_range_values() {
        let snap = NeuromodulatorSnapshot::from_scores(1, &[0.4, 0.3, 0.2, 1.0])
            .expect("in-range scores must construct");
        assert_eq!(snap.tick, 1);
        assert!((snap.dopamine - 0.4).abs() < 1e-6);
    }

    #[test]
    fn neuromodulator_snapshot_from_scores_rejects_out_of_range_value() {
        // tempo (scores[3]) outside documented [0.5, 2.0]
        let err = NeuromodulatorSnapshot::from_scores(1, &[0.4, 0.3, 0.2, 3.0])
            .expect_err("out-of-range tempo byte must fail construction, not just deserialization");
        assert_eq!(err.path, "tempo");
        assert_eq!(err.kind, crate::validation::ValidationKind::OutOfRange);
    }

    #[test]
    fn neuromodulator_snapshot_validate_rejects_out_of_range_value() {
        let mut snap = sample_neuromodulator_snapshot();
        snap.tempo = 3.0; // outside documented [0.5, 2.0]
        let err = snap
            .validate()
            .expect_err("out-of-range tempo must fail validation");
        assert_eq!(err.path, "tempo");
        assert_eq!(err.kind, crate::validation::ValidationKind::OutOfRange);
    }

    #[test]
    fn neuromodulator_snapshot_validate_rejects_non_finite_value() {
        let mut snap = sample_neuromodulator_snapshot();
        snap.dopamine = f32::NAN;
        let err = snap
            .validate()
            .expect_err("non-finite dopamine must fail validation");
        assert_eq!(err.path, "dopamine");
        assert_eq!(err.kind, crate::validation::ValidationKind::NonFinite);
    }

    #[test]
    fn neuromodulator_snapshot_deserialize_rejects_out_of_range_value() {
        let json = serde_json::json!({
            "tick": 1,
            "dopamine": 0.5,
            "cortisol": 0.5,
            "acetylcholine": 0.5,
            "tempo": 3.0
        });
        let result: Result<NeuromodulatorSnapshot, _> = serde_json::from_value(json);
        assert!(
            result.is_err(),
            "an out-of-range tempo must fail to deserialize"
        );
    }

    #[test]
    fn neuromodulator_snapshot_try_from_wire_rejects_non_finite_value() {
        // JSON has no NaN/Infinity literal, so this exercises the TryFrom
        // conversion that backs Deserialize directly, for wire formats
        // (e.g. bincode) that can represent a non-finite f32.
        let wire = NeuromodulatorSnapshotWire {
            tick: 1,
            dopamine: f32::NAN,
            cortisol: 0.5,
            acetylcholine: 0.5,
            tempo: 1.0,
        };
        assert!(
            NeuromodulatorSnapshot::try_from(wire).is_err(),
            "a non-finite dopamine must fail the TryFrom conversion"
        );
    }

    fn sample_trace_batch() -> TraceBatch {
        TraceBatch {
            session_id: "sess-1".into(),
            batch_id: 7,
            traces: vec![TraceData {
                channel_id: 3,
                trace_value: 0.25,
                last_spike_time: 11,
            }],
        }
    }

    #[test]
    fn ipc_aliases_are_the_same_types() {
        fn as_ipc_spike(batch: IpcSpikeBatch) -> SpikeBatch {
            batch
        }
        fn as_ipc_trace(batch: IpcTraceBatch) -> TraceBatch {
            batch
        }
        let _ = as_ipc_spike(sample_spike_batch());
        let _ = as_ipc_trace(sample_trace_batch());
    }

    #[test]
    fn spike_batch_json_keys_stay_stable() {
        let json = serde_json::to_value(sample_spike_batch()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "session_id": "sess-1",
                "batch_id": 7,
                "timestamp": 1_700_000_000,
                "spikes": [{
                    "channel": 3,
                    "time": 11,
                    "strength": 0.5
                }],
                "metadata": null
            })
        );
    }

    #[test]
    fn trace_batch_json_keys_stay_stable() {
        let json = serde_json::to_value(sample_trace_batch()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "session_id": "sess-1",
                "batch_id": 7,
                "traces": [{
                    "channel_id": 3,
                    "trace_value": 0.25,
                    "last_spike_time": 11
                }]
            })
        );
    }

    #[test]
    fn ipc_message_envelopes_keep_variant_names() {
        let spikes = serde_json::to_value(IpcMessage::Spikes(sample_spike_batch())).unwrap();
        let traces =
            serde_json::to_value(IpcMessage::EligibilityTraces(sample_trace_batch())).unwrap();
        assert!(spikes.get("Spikes").is_some());
        assert!(traces.get("EligibilityTraces").is_some());
        let decoded_spikes: IpcMessage = serde_json::from_value(spikes).unwrap();
        let decoded_traces: IpcMessage = serde_json::from_value(traces).unwrap();
        assert_eq!(decoded_spikes, IpcMessage::Spikes(sample_spike_batch()));
        assert_eq!(
            decoded_traces,
            IpcMessage::EligibilityTraces(sample_trace_batch())
        );
    }

    #[test]
    fn stimulus_batch_json_keys_stay_stable() {
        let json = serde_json::to_value(sample_stimulus_batch()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "session_id": "sess-1",
                "batch_id": 7,
                "timestamp": 1_700_000_000,
                "values": [0.5, 0.0, -0.25],
                "valid_mask": [true, false, true],
                "metadata": null
            })
        );
    }

    #[test]
    fn stimulus_batch_round_trips() {
        let batch = sample_stimulus_batch();
        let json = serde_json::to_value(&batch).unwrap();
        let decoded: StimulusBatch = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, batch);
    }

    #[test]
    fn stimulus_batch_default_has_no_mask_and_all_channels_valid() {
        let batch = StimulusBatch::default();
        assert!(batch.values.is_empty());
        assert!(batch.valid_mask.is_none());
    }

    #[test]
    fn stimulus_batch_validate_accepts_matching_or_absent_mask() {
        assert!(StimulusBatch::default().validate().is_ok());
        assert!(sample_stimulus_batch().validate().is_ok());
    }

    #[test]
    fn stimulus_batch_validate_rejects_mismatched_mask_length() {
        let batch = StimulusBatch {
            values: vec![0.0, 0.0, 0.0],
            valid_mask: Some(vec![true, false]),
            ..Default::default()
        };
        let err = batch
            .validate()
            .expect_err("mismatched mask must fail validation");
        assert_eq!(err.path, "valid_mask");
        assert_eq!(err.kind, crate::validation::ValidationKind::LengthMismatch);
        assert_eq!(
            err.actual,
            Some(crate::validation::ValidationMeasure::Count(2))
        );
        assert_eq!(
            err.limit,
            Some(crate::validation::ValidationMeasure::Count(3))
        );
    }

    #[test]
    fn stimulus_batch_deserialize_rejects_mismatched_mask_length() {
        let json = serde_json::json!({
            "session_id": null,
            "batch_id": 1,
            "timestamp": 0,
            "values": [0.0, 1.0],
            "valid_mask": [true],
            "metadata": null
        });
        let result: Result<StimulusBatch, _> = serde_json::from_value(json);
        assert!(
            result.is_err(),
            "a valid_mask shorter than values must fail to deserialize"
        );
    }

    #[test]
    fn stimulus_batch_invalid_channel_is_distinct_from_a_real_zero() {
        // Channel 1 is a genuine zero reading; channel 2 is missing/invalid and
        // its 0.0 placeholder must not be mistaken for a real reading.
        let batch = StimulusBatch {
            session_id: None,
            batch_id: 1,
            timestamp: 0,
            values: vec![1.0, 0.0, 0.0],
            valid_mask: Some(vec![true, true, false]),
            metadata: None,
        };
        let json = serde_json::to_value(&batch).unwrap();
        let decoded: StimulusBatch = serde_json::from_value(json).unwrap();
        let mask = decoded.valid_mask.expect("mask must survive round-trip");
        assert!(mask[1], "channel 1 is a valid, genuine zero reading");
        assert!(!mask[2], "channel 2 is invalid/missing, not a real zero");
    }

    #[test]
    fn ipc_message_stimuli_and_neuromodulators_keep_variant_names() {
        let stimuli = serde_json::to_value(IpcMessage::Stimuli(sample_stimulus_batch())).unwrap();
        let neuromods =
            serde_json::to_value(IpcMessage::Neuromodulators(sample_neuromodulator_snapshot()))
                .unwrap();
        assert!(stimuli.get("Stimuli").is_some());
        assert!(neuromods.get("Neuromodulators").is_some());
        let decoded_stimuli: IpcMessage = serde_json::from_value(stimuli).unwrap();
        let decoded_neuromods: IpcMessage = serde_json::from_value(neuromods).unwrap();
        assert_eq!(
            decoded_stimuli,
            IpcMessage::Stimuli(sample_stimulus_batch())
        );
        assert_eq!(
            decoded_neuromods,
            IpcMessage::Neuromodulators(sample_neuromodulator_snapshot())
        );
    }
}
