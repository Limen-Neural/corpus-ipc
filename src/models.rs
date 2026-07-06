// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data types that flow over the compute backend IPC wire.

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
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

impl RuntimeSnapshot {
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
}

/// Core message enum for Rust<->Julia cross-process communication.
///
/// Messages are separated into:
/// - Input from Rust to Julia (spikes, embeddings, config)
/// - Output from Julia to Rust (gradients, traces, training status)
/// - Control messages (shutdown, ping)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RuntimeMessage {
    // Input from Rust to Julia
    Spikes(SpikeBatch),
    Embeddings(EmbeddingBatch),
    Loss(f32),
    ConfigUpdate(ConfigPayload),

    // Output from Julia to Rust
    GradientUpdate(GradientBatch),
    EligibilityTraces(TraceBatch),
    TrainingComplete,

    // Control
    Shutdown,
    Ping,
}

/// Batch of spike events from compute processing.
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

/// Individual spike event with channel, timing, and strength.
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

/// Gradient update batch from Julia training algorithms.
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

/// Eligibility trace batch for credit assignment in spiking networks.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TraceBatch {
    /// Session ID for routing.
    pub session_id: String,
    /// Batch ID correlation.
    pub batch_id: u64,
    /// Eligibility trace data.
    pub traces: Vec<TraceData>,
}

/// Individual eligibility trace data.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TraceData {
    /// Channel or synapse identifier.
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
