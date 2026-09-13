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

use serde::{Deserialize, Serialize};

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

impl NeuromodulatorSnapshot {
    /// Parse from the 4 generic score floats in bytes `[72..88]` of a generic packet.
    pub fn from_scores(tick: i64, scores: &[f32; 4]) -> Self {
        Self {
            tick,
            dopamine: scores[0],
            cortisol: scores[1],
            acetylcholine: scores[2],
            tempo: scores[3],
        }
    }

    /// Check the documented ranges for each field.
    ///
    /// Fields are public and this type is now reachable via
    /// [`IpcMessage::Neuromodulators`], so a deserialized instance is not
    /// guaranteed to satisfy the documented ranges (`dopamine`, `cortisol`,
    /// `acetylcholine` in `[0, 1]`; `tempo` in `[0.5, 2.0]`) or even to be
    /// finite. Call this after building or deserializing one to check
    /// explicitly.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value, min, max) in [
            ("dopamine", self.dopamine, 0.0, 1.0),
            ("cortisol", self.cortisol, 0.0, 1.0),
            ("acetylcholine", self.acetylcholine, 0.0, 1.0),
            ("tempo", self.tempo, 0.5, 2.0),
        ] {
            if !value.is_finite() {
                return Err(format!(
                    "NeuromodulatorSnapshot: {name} is not finite: {value}"
                ));
            }
            if value < min || value > max {
                return Err(format!(
                    "NeuromodulatorSnapshot: {name} ({value}) out of documented range [{min}, {max}]"
                ));
            }
        }
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
    /// See [`NeuromodulatorSnapshot`] and [`NeuromodulatorSnapshot::validate`].
    Neuromodulators(NeuromodulatorSnapshot),
    Loss(f32),
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
/// [`StimulusBatch::validate`] explicitly after constructing one directly in
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

impl StimulusBatch {
    /// Check the `valid_mask`-length invariant documented on this type.
    ///
    /// Returns `Err` describing the mismatch when `valid_mask` is `Some` and
    /// its length differs from `values.len()`. Returns `Ok(())` when
    /// `valid_mask` is `None` or already matches.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(mask) = &self.valid_mask
            && mask.len() != self.values.len()
        {
            return Err(format!(
                "StimulusBatch: valid_mask length ({}) must match values length ({})",
                mask.len(),
                self.values.len()
            ));
        }
        Ok(())
    }
}

/// Deserialization-only shadow of [`StimulusBatch`] with the identical wire
/// shape. `StimulusBatch`'s real `Deserialize` impl (below) goes through
/// this type and [`StimulusBatch::validate`] so a malformed `valid_mask`
/// length is rejected at deserialization instead of producing an
/// inconsistent instance.
#[derive(Deserialize)]
struct StimulusBatchWire {
    session_id: Option<String>,
    batch_id: u64,
    timestamp: u64,
    values: Vec<f32>,
    valid_mask: Option<Vec<bool>>,
    metadata: Option<BatchMetadata>,
}

impl TryFrom<StimulusBatchWire> for StimulusBatch {
    type Error = String;

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

/// Explicit IPC-domain name for [`SpikeBatch`].
///
/// Same type and same wire format. Prefer this alias in new code that sits
/// next to SynapticDistill training types.
pub type IpcSpikeBatch = SpikeBatch;

/// Individual spike event with channel, timing, and strength.
///
/// An element of an IPC [`SpikeBatch`], not a SynapticDistill training row.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct SpikeEvent {
    /// Compute channel or channel identifier.
    pub channel: u16,
    /// Spike timestamp (relative or absolute).
    pub time: u32,
    /// Spike strength or amplitude.
    pub strength: f32,
}

/// Batch of embeddings for projector and transformer components.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
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

/// Gradient update batch from external training or optimization algorithms.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GradientBatch {
    /// Session ID for routing back to correct experiment.
    pub session_id: String,
    /// Batch ID correlation with original input.
    pub batch_id: u64,
    /// Individual gradient updates.
    pub gradients: Vec<GradientUpdate>,
}

/// Individual gradient update for a specific layer or parameter.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GradientUpdate {
    /// Target layer identifier.
    pub layer_id: String,
    /// Gradient values (flattened or sparse representation).
    pub gradients: Vec<f32>,
    /// Optional eligibility trace for E-prop algorithms.
    pub eligibility_trace: Option<Vec<f32>>,
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

/// Individual eligibility trace data on the IPC wire.
///
/// An element of an IPC [`TraceBatch`], not a SynapticDistill training trace.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TraceData {
    /// Channel or synapse identifier.
    #[serde(alias = "neuron_id")]
    pub channel_id: u16,
    /// Trace value (decay-modulated spike history).
    pub trace_value: f32,
    /// Timestamp of last contributing spike.
    pub last_spike_time: u32,
}

/// Configuration payload for runtime parameter updates.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConfigPayload {
    /// Target session (None = global).
    pub session_id: Option<String>,
    /// Configuration key-value pairs.
    pub config: std::collections::HashMap<String, ConfigValue>,
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
    Float(f32),

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
    String(String),

    /// Boolean value.
    Boolean(bool),
    FloatArray(Vec<f32>),
}

/// Optional batch metadata for debugging and monitoring.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BatchMetadata {
    /// Processing latency in nanoseconds.
    pub processing_latency_ns: Option<u64>,
    /// Source identifier (for example, "encoder", "compute_layer_2").
    pub source: Option<String>,
    /// Additional metadata fields.
    pub custom: std::collections::HashMap<String, String>,
}

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
    fn neuromodulator_snapshot_validate_rejects_out_of_range_value() {
        let mut snap = sample_neuromodulator_snapshot();
        snap.tempo = 3.0; // outside documented [0.5, 2.0]
        let err = snap
            .validate()
            .expect_err("out-of-range tempo must fail validation");
        assert!(err.contains("tempo"), "error should name the field: {err}");
    }

    #[test]
    fn neuromodulator_snapshot_validate_rejects_non_finite_value() {
        let mut snap = sample_neuromodulator_snapshot();
        snap.dopamine = f32::NAN;
        let err = snap
            .validate()
            .expect_err("non-finite dopamine must fail validation");
        assert!(
            err.contains("dopamine"),
            "error should name the field: {err}"
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
        assert!(
            err.contains('3'),
            "error should mention values length: {err}"
        );
        assert!(err.contains('2'), "error should mention mask length: {err}");
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
