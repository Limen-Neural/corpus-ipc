// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared corpus definitions for the wire-v1 golden-vector tests
//! (`wire_v1_vectors.rs`, `wire_v1_regenerate.rs`; LIM-1446 / RM-1330).
//!
//! Not a test target itself — `tests/` subdirectories are not compiled as
//! integration tests; each target pulls this in with `mod shared;`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use corpus_ipc::{
    BatchMetadata, ConfigPayload, ConfigValue, EmbeddingBatch, GradientBatch, GradientUpdate,
    IpcMessage, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch, TraceBatch,
    TraceData,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const REGENERATE_ENV: &str = "REGENERATE_WIRE_V1";

/// Regeneration command recorded in `manifest.json` and in the test docs.
pub const REGENERATE_COMMAND: &str =
    "REGENERATE_WIRE_V1=1 cargo test --locked --test wire_v1_regenerate";

/// Every `IpcMessage` variant that must have a positive vector. When a new
/// variant is added to the enum, add a `positive_vectors` entry and its name
/// here; `positive_corpus_covers_every_variant` fails on any omission.
pub const EXPECTED_VARIANTS: &[&str] = &[
    "Spikes",
    "Embeddings",
    "Stimuli",
    "Neuromodulators",
    "Loss",
    "ConfigUpdate",
    "GradientUpdate",
    "EligibilityTraces",
    "TrainingComplete",
    "Shutdown",
    "Ping",
];

#[derive(Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub encoding_profile: String,
    pub generator: String,
    pub fixtures: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub file: String,
    pub sha256: String,
    pub kind: String,
    #[serde(default)]
    pub variant: Option<String>,
    pub expect: String,
}

pub fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/wire-v1")
}

pub fn load_manifest() -> Manifest {
    let path = fixture_root().join("manifest.json");
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read manifest: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse manifest: {e}"))
}

pub fn fixture_bytes(entry: &ManifestEntry) -> Vec<u8> {
    std::fs::read(fixture_root().join(&entry.file))
        .unwrap_or_else(|e| panic!("read {}: {e}", entry.file))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// A positive vector: file name, `IpcMessage` variant name, message.
pub type Vector = (&'static str, &'static str, IpcMessage);

/// Positive (canonical, generated) vectors. Split into per-group builders so
/// no single function grows unwieldy.
pub fn positive_vectors() -> Vec<Vector> {
    [
        control_vectors(),
        spike_vectors(),
        input_vectors(),
        config_vectors(),
        output_vectors(),
    ]
    .concat()
}

fn control_vectors() -> Vec<Vector> {
    vec![
        ("ping", "Ping", IpcMessage::Ping),
        ("shutdown", "Shutdown", IpcMessage::Shutdown),
        (
            "training_complete",
            "TrainingComplete",
            IpcMessage::TrainingComplete,
        ),
        ("loss", "Loss", IpcMessage::Loss(0.1)),
    ]
}

fn spike_vectors() -> Vec<Vector> {
    vec![
        ("spikes", "Spikes", IpcMessage::Spikes(sample_spikes())),
        ("boundaries", "Spikes", boundaries_spikes_message()),
        ("empty_spikes", "Spikes", empty_spikes_message()),
        ("unicode_metadata", "Spikes", unicode_spikes_message()),
    ]
}

fn sample_spikes() -> SpikeBatch {
    SpikeBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        spikes: vec![
            SpikeEvent {
                channel: 3,
                time: 11,
                strength: 0.5,
            },
            SpikeEvent {
                channel: 5,
                time: 17,
                strength: 1.25,
            },
        ],
        metadata: Some(BatchMetadata {
            processing_latency_ns: Some(128),
            source: Some("encoder".into()),
            custom: HashMap::new(),
        }),
    }
}

/// Representable boundaries: integer extremes that stay exact on the wire
/// (Rust `u*`/`i*` encode as JSON integers, not f32).
fn boundaries_spikes_message() -> IpcMessage {
    IpcMessage::Spikes(SpikeBatch {
        session_id: None,
        batch_id: u64::MAX,
        timestamp: u64::MAX,
        spikes: vec![SpikeEvent {
            channel: u16::MAX,
            time: u32::MAX,
            strength: f32::MAX,
        }],
        metadata: None,
    })
}

fn empty_spikes_message() -> IpcMessage {
    IpcMessage::Spikes(SpikeBatch {
        session_id: None,
        batch_id: 0,
        timestamp: 0,
        spikes: vec![],
        metadata: None,
    })
}

fn unicode_spikes_message() -> IpcMessage {
    let mut custom = HashMap::new();
    custom.insert("café".into(), "naïve-Ω".into());
    custom.insert("日本語".into(), "spike-viz".into());
    IpcMessage::Spikes(SpikeBatch {
        session_id: Some("séssion-日本語".into()),
        batch_id: 55,
        timestamp: 1_700_002_000,
        spikes: vec![SpikeEvent {
            channel: 1,
            time: 2,
            strength: 0.5,
        }],
        metadata: Some(BatchMetadata {
            processing_latency_ns: Some(42),
            source: Some("encoder-Ω".into()),
            custom,
        }),
    })
}

fn input_vectors() -> Vec<Vector> {
    vec![
        (
            "embeddings",
            "Embeddings",
            IpcMessage::Embeddings(EmbeddingBatch {
                session_id: Some("sess-1".into()),
                batch_id: 9,
                embedding: vec![0.125, -0.5, 1.0],
                sequence_length: 3,
            }),
        ),
        ("stimuli_masked", "Stimuli", masked_stimuli_message()),
        ("stimuli_unmasked", "Stimuli", unmasked_stimuli_message()),
        (
            "neuromodulators",
            "Neuromodulators",
            IpcMessage::Neuromodulators(NeuromodulatorSnapshot {
                tick: 4_096,
                dopamine: 0.8,
                cortisol: 0.1,
                acetylcholine: 0.6,
                tempo: 1.0,
            }),
        ),
    ]
}

fn masked_stimuli_message() -> IpcMessage {
    IpcMessage::Stimuli(StimulusBatch {
        session_id: Some("sess-1".into()),
        batch_id: 11,
        timestamp: 1_700_000_500,
        values: vec![0.75, 0.0, -0.25],
        // Channel 1 is missing this tick: values[1] is a placeholder.
        valid_mask: Some(vec![true, false, true]),
        metadata: None,
    })
}

fn unmasked_stimuli_message() -> IpcMessage {
    IpcMessage::Stimuli(StimulusBatch {
        session_id: None,
        batch_id: 12,
        timestamp: 1_700_001_000,
        values: vec![0.5, -0.5],
        valid_mask: None,
        metadata: None,
    })
}

fn config_vectors() -> Vec<Vector> {
    vec![
        ("config_update", "ConfigUpdate", config_update_message()),
        ("empty_collections", "ConfigUpdate", empty_config_message()),
        ("signed_zero", "ConfigUpdate", signed_zero_message()),
    ]
}

fn config_update_message() -> IpcMessage {
    let mut config = HashMap::new();
    config.insert("alpha".into(), ConfigValue::Boolean(true));
    config.insert("depth".into(), ConfigValue::Float(0.25));
    config.insert("mode".into(), ConfigValue::String("replay".into()));
    config.insert("tau".into(), ConfigValue::FloatArray(vec![0.1, 0.2, 0.3]));
    config.insert("zeta".into(), ConfigValue::Float(7.0));
    IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: Some("sess-1".into()),
        config,
    })
}

/// Legal empty collections must stay legal on the wire.
fn empty_config_message() -> IpcMessage {
    IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: None,
        config: HashMap::new(),
    })
}

fn signed_zero_message() -> IpcMessage {
    let mut config = HashMap::new();
    config.insert("neg".into(), ConfigValue::Float(-0.0));
    config.insert("pos".into(), ConfigValue::Float(0.0));
    IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: None,
        config,
    })
}

fn output_vectors() -> Vec<Vector> {
    vec![
        (
            "gradient_update",
            "GradientUpdate",
            IpcMessage::GradientUpdate(GradientBatch {
                session_id: "sess-1".into(),
                batch_id: 21,
                gradients: vec![
                    GradientUpdate {
                        layer_id: "layer-0".into(),
                        gradients: vec![0.01, -0.02],
                        eligibility_trace: Some(vec![0.5, 0.25]),
                    },
                    GradientUpdate {
                        layer_id: "layer-1".into(),
                        gradients: vec![0.5],
                        eligibility_trace: None,
                    },
                ],
            }),
        ),
        (
            "eligibility_traces",
            "EligibilityTraces",
            IpcMessage::EligibilityTraces(TraceBatch {
                session_id: "sess-1".into(),
                batch_id: 34,
                traces: vec![
                    TraceData {
                        channel_id: 5,
                        trace_value: 0.75,
                        last_spike_time: 1_000,
                    },
                    TraceData {
                        channel_id: 9,
                        trace_value: 0.5,
                        last_spike_time: 2_000,
                    },
                ],
            }),
        ),
    ]
}

/// Hand-maintained vectors whose expected decode is a full `IpcMessage`.
/// These are *not* regenerated: legacy unversioned JSON, the `neuron_id`
/// field alias, and the documented Integer -> Float normalization.
pub fn decode_vectors() -> Vec<(&'static str, IpcMessage)> {
    vec![
        ("legacy_unversioned_ping", IpcMessage::Ping),
        (
            "legacy_unversioned_spikes",
            IpcMessage::Spikes(SpikeBatch {
                session_id: Some("sess-1".into()),
                batch_id: 7,
                timestamp: 1_700_000_000,
                spikes: vec![SpikeEvent {
                    channel: 3,
                    time: 11,
                    strength: 0.5,
                }],
                metadata: None,
            }),
        ),
        ("alias_neuron_id_traces", alias_traces_message()),
        ("normalization_integer_config", normalization_message()),
    ]
}

/// The wire field is `channel_id`; `neuron_id` is a supported
/// deserialization alias for older producers.
fn alias_traces_message() -> IpcMessage {
    IpcMessage::EligibilityTraces(TraceBatch {
        session_id: "sess-1".into(),
        batch_id: 34,
        traces: vec![TraceData {
            channel_id: 5,
            trace_value: 0.75,
            last_spike_time: 1_000,
        }],
    })
}

/// Normalization: JSON integers decode as `ConfigValue::Float`. 42 ->
/// `Float(42.0)`; 16_777_217 (2^24 + 1) is not representable in f32 and
/// decodes as `Float(16_777_216.0)`.
fn normalization_message() -> IpcMessage {
    IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: Some("sess-1".into()),
        config: HashMap::from([
            ("n".to_string(), ConfigValue::Float(42.0)),
            ("big".to_string(), ConfigValue::Float(16_777_216.0)),
        ]),
    })
}
