// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wire-v1 golden-vector corpus tests (LIM-1446 / RM-1330).
//!
//! `test-vectors/wire-v1/` is the published golden corpus for the canonical
//! wire-v1 encoding (`encode_canonical_ipc_message`, `docs/wire-encoding.md`).
//!
//! - **Positive** vectors are *generated* by the canonical encoder and must
//!   stay byte-stable: the file bytes must equal a fresh encode of the same
//!   message, and decode/re-encode must round-trip to identical bytes.
//! - **Legacy**, **alias**, and **normalization** vectors are hand-maintained
//!   decode targets: they record the expected decoded `IpcMessage` (including
//!   the documented `ConfigValue::Integer` -> `ConfigValue::Float`
//!   normalization and `neuron_id` -> `channel_id` alias). They must never be
//!   regenerated through the canonical encoder.
//! - **Negative** vectors record the expected failure class (too old, too new,
//!   payload error, JSON error). Unsupported-version vectors prove the version
//!   check runs before payload interpretation.
//!
//! Fixture drift on a positive vector means either the fixture or the encoder
//! changed. Regenerate intentionally with:
//!
//! ```bash
//! REGENERATE_WIRE_V1=1 cargo test --locked --test wire_v1_vectors regenerate_fixtures
//! ```
//!
//! which rewrites the positive `*.json` files and recomputes every SHA-256 in
//! `manifest.json`. Inspect the diff before committing.

use std::collections::HashMap;
use std::path::PathBuf;

use corpus_ipc::{
    BatchMetadata, CompatibilityError, ConfigPayload, ConfigValue, EmbeddingBatch, EnvelopeError,
    GradientBatch, GradientUpdate, IpcMessage, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent,
    StimulusBatch, TraceBatch, TraceData, WireCompatibility, decode_ipc_message_json,
    encode_canonical_ipc_message,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const REGENERATE_ENV: &str = "REGENERATE_WIRE_V1";

/// Every `IpcMessage` variant that must have a positive vector. When a new
/// variant is added to the enum, add a `positive_vectors` entry and its name
/// here; `positive_corpus_covers_every_variant` fails on any omission.
const EXPECTED_VARIANTS: &[&str] = &[
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
struct Manifest {
    schema_version: u32,
    encoding_profile: String,
    generator: String,
    fixtures: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    name: String,
    file: String,
    sha256: String,
    kind: String,
    #[serde(default)]
    variant: Option<String>,
    expect: String,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/wire-v1")
}

fn load_manifest() -> Manifest {
    let path = fixture_root().join("manifest.json");
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read manifest: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse manifest: {e}"))
}

fn fixture_bytes(entry: &ManifestEntry) -> Vec<u8> {
    std::fs::read(fixture_root().join(&entry.file))
        .unwrap_or_else(|e| panic!("read {}: {e}", entry.file))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

type Vector = (&'static str, &'static str, IpcMessage);

/// Positive (canonical, generated) vectors: file name, `IpcMessage` variant
/// name, and the message the canonical encoder emits bytes for. Split into
/// per-group builders so no single function grows unwieldy.
fn positive_vectors() -> Vec<Vector> {
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
        (
            "spikes",
            "Spikes",
            IpcMessage::Spikes(SpikeBatch {
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
            }),
        ),
        (
            "boundaries",
            "Spikes",
            // Representable boundaries: integer extremes that stay exact on
            // the wire (Rust `u*`/`i*` encode as JSON integers, not f32).
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
            }),
        ),
        (
            "empty_spikes",
            "Spikes",
            IpcMessage::Spikes(SpikeBatch {
                session_id: None,
                batch_id: 0,
                timestamp: 0,
                spikes: vec![],
                metadata: None,
            }),
        ),
        ("unicode_metadata", "Spikes", unicode_spikes_message()),
    ]
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
        (
            "stimuli_masked",
            "Stimuli",
            IpcMessage::Stimuli(StimulusBatch {
                session_id: Some("sess-1".into()),
                batch_id: 11,
                timestamp: 1_700_000_500,
                values: vec![0.75, 0.0, -0.25],
                // Channel 1 is missing this tick: values[1] is a placeholder.
                valid_mask: Some(vec![true, false, true]),
                metadata: None,
            }),
        ),
        (
            "stimuli_unmasked",
            "Stimuli",
            IpcMessage::Stimuli(StimulusBatch {
                session_id: None,
                batch_id: 12,
                timestamp: 1_700_001_000,
                values: vec![0.5, -0.5],
                valid_mask: None,
                metadata: None,
            }),
        ),
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

fn config_vectors() -> Vec<Vector> {
    vec![
        ("config_update", "ConfigUpdate", config_update_message()),
        (
            "empty_collections",
            "ConfigUpdate",
            // Legal empty collections must stay legal on the wire.
            IpcMessage::ConfigUpdate(ConfigPayload {
                session_id: None,
                config: HashMap::new(),
            }),
        ),
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
fn decode_vectors() -> Vec<(&'static str, IpcMessage)> {
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
        (
            // The wire field is `channel_id`; `neuron_id` is a supported
            // deserialization alias for older producers.
            "alias_neuron_id_traces",
            IpcMessage::EligibilityTraces(TraceBatch {
                session_id: "sess-1".into(),
                batch_id: 34,
                traces: vec![TraceData {
                    channel_id: 5,
                    trace_value: 0.75,
                    last_spike_time: 1_000,
                }],
            }),
        ),
        (
            // Normalization: JSON integers decode as ConfigValue::Float.
            // 42 -> Float(42.0); 16_777_217 (2^24 + 1) is not representable
            // in f32 and decodes as Float(16_777_216.0).
            "normalization_integer_config",
            IpcMessage::ConfigUpdate(ConfigPayload {
                session_id: Some("sess-1".into()),
                config: HashMap::from([
                    ("n".to_string(), ConfigValue::Float(42.0)),
                    ("big".to_string(), ConfigValue::Float(16_777_216.0)),
                ]),
            }),
        ),
    ]
}

#[test]
fn manifest_is_well_formed_and_digests_match() {
    let manifest = load_manifest();
    assert_eq!(manifest.schema_version, 1);
    assert!(
        manifest.encoding_profile.contains("wire v1"),
        "profile must name the wire-v1 encoding: {}",
        manifest.encoding_profile
    );
    assert!(
        manifest.generator.contains("wire_v1_vectors"),
        "manifest must record the regeneration command"
    );
    let mut seen = std::collections::HashSet::new();
    for entry in &manifest.fixtures {
        assert!(
            seen.insert(entry.name.as_str()),
            "duplicate fixture name {}",
            entry.name
        );
        let bytes = fixture_bytes(entry);
        assert_eq!(
            sha256_hex(&bytes),
            entry.sha256,
            "sha256 drift on {}",
            entry.name
        );
    }
}

#[test]
fn positive_corpus_covers_every_variant() {
    let manifest = load_manifest();
    let covered: std::collections::HashSet<&str> = manifest
        .fixtures
        .iter()
        .filter(|f| f.kind == "positive")
        .filter_map(|f| f.variant.as_deref())
        .collect();
    for variant in EXPECTED_VARIANTS {
        assert!(
            covered.contains(variant),
            "missing positive vector for IpcMessage::{variant}"
        );
    }
    // Every constructor entry is distinct and names a real variant.
    let mut files = std::collections::HashSet::new();
    for (name, variant, _) in positive_vectors() {
        assert!(files.insert(name), "duplicate positive name {name}");
        assert!(
            EXPECTED_VARIANTS.contains(&variant),
            "{name} names unknown variant {variant}"
        );
    }
}

#[test]
fn positive_vectors_are_byte_stable_and_round_trip() {
    let manifest = load_manifest();
    let by_name: HashMap<&str, &ManifestEntry> = manifest
        .fixtures
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    for (name, variant, message) in positive_vectors() {
        let entry = by_name
            .get(name)
            .unwrap_or_else(|| panic!("manifest lacks positive {name}"));
        assert_eq!(entry.kind, "positive", "{name} kind");
        assert_eq!(entry.variant.as_deref(), Some(variant), "{name} variant");
        assert_eq!(entry.expect, "decode_ok", "{name} expectation");

        let on_disk = fixture_bytes(entry);
        // Byte-stable regeneration: file bytes equal a fresh canonical encode.
        let regenerated =
            encode_canonical_ipc_message(&message).unwrap_or_else(|e| panic!("encode {name}: {e}"));
        assert_eq!(
            on_disk, regenerated,
            "{name} drifted; regenerate with REGENERATE_WIRE_V1=1 \
             cargo test --test wire_v1_vectors regenerate_fixtures"
        );
        // Canonical decode/re-encode equivalence.
        let decoded =
            decode_ipc_message_json(&on_disk).unwrap_or_else(|e| panic!("decode {name}: {e}"));
        assert_eq!(
            decoded, message,
            "{name} decoded value mismatch (note: ConfigValue::Integer normalizes \
             to Float on decode; keep positives inside the normalized domain)"
        );
        let reencoded = encode_canonical_ipc_message(&decoded).unwrap();
        assert_eq!(reencoded, on_disk, "{name} re-encode not identical");
    }
}

#[test]
fn hand_maintained_vectors_decode_to_expected_messages() {
    let manifest = load_manifest();
    let by_name: HashMap<&str, &ManifestEntry> = manifest
        .fixtures
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    for (name, expected) in decode_vectors() {
        let entry = by_name
            .get(name)
            .unwrap_or_else(|| panic!("manifest lacks {name}"));
        assert!(
            matches!(entry.kind.as_str(), "legacy" | "alias" | "normalization"),
            "{name} kind {}",
            entry.kind
        );
        assert_eq!(entry.expect, "decode_ok", "{name} expectation");
        let decoded = decode_ipc_message_json(&fixture_bytes(entry))
            .unwrap_or_else(|e| panic!("decode {name}: {e}"));
        assert_eq!(decoded, expected, "{name} decoded value mismatch");
    }
}

#[test]
fn normalization_vector_documents_integer_to_float_loss() {
    // Explicitly record the non-lossless decode: the on-wire `big` integer is
    // 16_777_217 but decodes as Float(16_777_216.0) (2^24), the closest f32.
    let manifest = load_manifest();
    let entry = manifest
        .fixtures
        .iter()
        .find(|f| f.name == "normalization_integer_config")
        .expect("manifest entry");
    let bytes = fixture_bytes(entry);
    assert!(
        String::from_utf8_lossy(&bytes).contains("16777217"),
        "fixture must keep the literal 2^24+1 input"
    );
    let IpcMessage::ConfigUpdate(payload) = decode_ipc_message_json(&bytes).unwrap() else {
        panic!("expected ConfigUpdate");
    };
    assert_eq!(payload.config.get("n"), Some(&ConfigValue::Float(42.0)));
    match payload.config.get("big") {
        Some(ConfigValue::Float(v)) => {
            assert_eq!(*v, 16_777_216.0);
            assert_ne!(*v as u64, 16_777_217, "precision loss is the contract");
        }
        other => panic!("expected Float after normalization, got {other:?}"),
    }
}

#[test]
fn negative_vectors_fail_with_expected_error_class() {
    let manifest = load_manifest();
    let mut negatives = 0;
    for entry in manifest.fixtures.iter().filter(|f| f.kind == "negative") {
        negatives += 1;
        let err = decode_ipc_message_json(&fixture_bytes(entry))
            .expect_err(&format!("{} must fail", entry.name));
        match entry.expect.as_str() {
            "too_old" => assert!(
                matches!(
                    err,
                    EnvelopeError::Compatibility(CompatibilityError::TooOld { .. })
                ),
                "{}: expected TooOld, got {err}",
                entry.name
            ),
            "too_new" => assert!(
                matches!(
                    err,
                    EnvelopeError::Compatibility(CompatibilityError::TooNew { .. })
                ),
                "{}: expected TooNew, got {err}",
                entry.name
            ),
            "payload_error" => assert!(
                matches!(err, EnvelopeError::Payload(_)),
                "{}: expected payload error, got {err}",
                entry.name
            ),
            "json_error" => assert!(
                matches!(err, EnvelopeError::Json(_)),
                "{}: expected JSON error, got {err}",
                entry.name
            ),
            other => panic!("{}: unknown expectation {other}", entry.name),
        }
    }
    assert!(negatives >= 5, "corpus needs negative coverage");
}

#[test]
fn unsupported_version_is_rejected_before_payload_interpretation() {
    // v2 envelope whose payload is *not valid JSON-typed IpcMessage at all*:
    // if the version check did not run first this would surface a payload
    // error instead of TooNew.
    let manifest = load_manifest();
    let entry = manifest
        .fixtures
        .iter()
        .find(|f| f.name == "v2_too_new_malformed_payload")
        .expect("manifest entry");
    let err = decode_ipc_message_json(&fixture_bytes(entry)).unwrap_err();
    assert!(
        matches!(
            err,
            EnvelopeError::Compatibility(CompatibilityError::TooNew { found, .. })
                if found == WireCompatibility::CURRENT + 1
        ),
        "version must be checked before payload use, got {err}"
    );
}

/// Regenerate the corpus. Gated on `REGENERATE_WIRE_V1=1` so it never runs
/// accidentally in CI; when run it rewrites the positive fixture files from
/// `encode_canonical_ipc_message` and recomputes every SHA-256 in the
/// manifest. Hand-maintained fixture *content* is never touched.
#[test]
fn regenerate_fixtures() {
    if std::env::var_os(REGENERATE_ENV).is_none() {
        eprintln!("skipping regeneration: set {REGENERATE_ENV}=1 to enable");
        return;
    }
    let root = fixture_root();
    let manifest = load_manifest();
    let positive: HashMap<&str, IpcMessage> = positive_vectors()
        .into_iter()
        .map(|(n, _, m)| (n, m))
        .collect();

    let mut out = String::from(
        "{\n  \"schema_version\": 1,\n  \"encoding_profile\": \"corpus-ipc wire v1 \
         canonical JSON (docs/wire-encoding.md)\",\n  \"generator\": \"REGENERATE_WIRE_V1=1 \
         cargo test --locked --test wire_v1_vectors regenerate_fixtures\",\n  \"fixtures\": [\n",
    );
    let mut entries = Vec::new();
    for entry in &manifest.fixtures {
        let path = root.join(&entry.file);
        if entry.kind == "positive" {
            let message = positive
                .get(entry.name.as_str())
                .unwrap_or_else(|| panic!("no constructor for positive {}", entry.name));
            let bytes = encode_canonical_ipc_message(message)
                .unwrap_or_else(|e| panic!("encode {}: {e}", entry.name));
            std::fs::write(&path, &bytes)
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let variant = entry
            .variant
            .as_ref()
            .map(|v| format!(",\n      \"variant\": \"{v}\""))
            .unwrap_or_default();
        entries.push(format!(
            "    {{\n      \"name\": \"{}\",\n      \"file\": \"{}\",\n      \"sha256\": \
             \"{}\",\n      \"kind\": \"{}\"{},\n      \"expect\": \"{}\"\n    }}",
            entry.name,
            entry.file,
            sha256_hex(&bytes),
            entry.kind,
            variant,
            entry.expect
        ));
    }
    out.push_str(&entries.join(",\n"));
    out.push_str("\n  ]\n}\n");
    std::fs::write(root.join("manifest.json"), out).expect("write manifest.json");
    eprintln!(
        "regenerated {} fixtures under {}",
        entries.len(),
        root.display()
    );
}

/// Keep fixture files readable: every positive file must be exactly the
/// canonical compact encoding (no trailing newline, no whitespace).
#[test]
fn positive_files_are_compact_canonical_json() {
    let manifest = load_manifest();
    for entry in manifest.fixtures.iter().filter(|f| f.kind == "positive") {
        let bytes = fixture_bytes(entry);
        assert!(
            !bytes.ends_with(b"\n"),
            "{}: canonical JSON has no trailing newline",
            entry.name
        );
        let text = String::from_utf8(bytes).expect("utf8");
        assert_eq!(
            text,
            text.trim(),
            "{}: no surrounding whitespace",
            entry.name
        );
    }
}
