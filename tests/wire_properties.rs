// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible property tests for validated wire-v1 messages (LIM-1331).
//!
//! Generators are bounded (finite `f32`s, short strings/vectors, unique ids
//! where `Validate` requires them) so every generated [`IpcMessage`] satisfies
//! `ProtocolLimits::DEFAULT` and can go through
//! [`encode_canonical_ipc_message`]. Failures are reproducible: proptest emits
//! the failing seed and persists minimized counterexamples under
//! `tests/wire_properties.proptest-regressions` (set `PROPTEST_CASES` to
//! scale; default here is 64 cases per property for a fast CI run).
//!
//! Scope notes:
//! - Exact enum equality is asserted only over the losslessly representable
//!   domain. `ConfigValue::Integer` normalizes to `ConfigValue::Float` through
//!   JSON; that is covered by dedicated normalization properties, not by the
//!   identity property's generator.
//! - Duplicate-key rejection is byte-level: `serde_json::Value` has already
//!   deduplicated keys by the time `decode_ipc_message_value` sees it, so
//!   duplicate coverage goes through the byte entry point only.

use std::collections::HashMap;

use corpus_ipc::{
    BatchMetadata, ConfigPayload, ConfigValue, EmbeddingBatch, GradientBatch, GradientUpdate,
    IpcMessage, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch, TraceBatch,
    TraceData, decode_ipc_message_json, encode_canonical_ipc_message,
};
use proptest::prelude::*;
use serde_json::Value;

/// Small, fast bounds for generated collections. All are far below
/// `ProtocolLimits::DEFAULT` so validation always passes.
const MAX_ITEMS: usize = 8;
const MAX_STRING: usize = 32;

fn arb_f32() -> impl Strategy<Value = f32> {
    // Finite f32s only: `Validate` rejects NaN/inf, and the canonical encoder
    // is exercised separately for those.
    (-1.0e30_f32..1.0e30).prop_map(|v| if v.is_finite() { v } else { 0.0 })
}

fn arb_unit_f32() -> impl Strategy<Value = f32> {
    0.0_f32..=1.0
}

fn arb_tempo() -> impl Strategy<Value = f32> {
    0.5_f32..=2.0
}

fn arb_string() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_.-]{0,32}".prop_map(|mut s| {
        s.truncate(MAX_STRING);
        s
    })
}

fn arb_nonempty_string() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_.-]{1,32}"
}

fn arb_metadata() -> impl Strategy<Value = BatchMetadata> {
    (
        proptest::option::of(any::<u64>()),
        proptest::option::of(arb_string()),
        proptest::collection::hash_map(arb_nonempty_string(), arb_string(), 0..=4),
    )
        .prop_map(|(processing_latency_ns, source, custom)| BatchMetadata {
            processing_latency_ns,
            source,
            custom,
        })
}

fn arb_spike_batch() -> impl Strategy<Value = SpikeBatch> {
    (
        proptest::option::of(arb_string()),
        any::<u64>(),
        any::<u64>(),
        proptest::collection::vec((any::<u16>(), any::<u32>(), arb_f32()), 0..=MAX_ITEMS),
        proptest::option::of(arb_metadata()),
    )
        .prop_map(
            |(session_id, batch_id, timestamp, spikes, metadata)| SpikeBatch {
                session_id,
                batch_id,
                timestamp,
                spikes: spikes
                    .into_iter()
                    .map(|(channel, time, strength)| SpikeEvent {
                        channel,
                        time,
                        strength,
                    })
                    .collect(),
                metadata,
            },
        )
}

fn arb_embedding_batch() -> impl Strategy<Value = EmbeddingBatch> {
    (
        proptest::option::of(arb_string()),
        any::<u64>(),
        proptest::collection::vec(arb_f32(), 0..=MAX_ITEMS),
        0_usize..64,
    )
        .prop_map(
            |(session_id, batch_id, embedding, sequence_length)| EmbeddingBatch {
                session_id,
                batch_id,
                embedding,
                sequence_length,
            },
        )
}

fn arb_stimulus_batch() -> impl Strategy<Value = StimulusBatch> {
    (
        proptest::option::of(arb_string()),
        any::<u64>(),
        any::<u64>(),
        proptest::collection::vec(arb_f32(), 0..=MAX_ITEMS),
        proptest::option::of(arb_metadata()),
    )
        .prop_flat_map(|(session_id, batch_id, timestamp, values, metadata)| {
            let len = values.len();
            (
                Just(session_id),
                Just(batch_id),
                Just(timestamp),
                Just(values),
                proptest::option::of(proptest::collection::vec(any::<bool>(), len..=len)),
                Just(metadata),
            )
        })
        .prop_map(
            |(session_id, batch_id, timestamp, values, valid_mask, metadata)| StimulusBatch {
                session_id,
                batch_id,
                timestamp,
                values,
                valid_mask,
                metadata,
            },
        )
}

fn arb_neuromodulator_snapshot() -> impl Strategy<Value = NeuromodulatorSnapshot> {
    (
        any::<i64>(),
        arb_unit_f32(),
        arb_unit_f32(),
        arb_unit_f32(),
        arb_tempo(),
    )
        .prop_map(
            |(tick, dopamine, cortisol, acetylcholine, tempo)| NeuromodulatorSnapshot {
                tick,
                dopamine,
                cortisol,
                acetylcholine,
                tempo,
            },
        )
}

/// Losslessly representable `ConfigValue` domain: no `Integer` (it normalizes
/// to `Float` on the wire — see the normalization properties below).
fn arb_config_value() -> impl Strategy<Value = ConfigValue> {
    prop_oneof![
        arb_f32().prop_map(ConfigValue::Float),
        arb_string().prop_map(ConfigValue::String),
        any::<bool>().prop_map(ConfigValue::Boolean),
        proptest::collection::vec(arb_f32(), 0..=MAX_ITEMS).prop_map(ConfigValue::FloatArray),
    ]
}

fn arb_config_payload() -> impl Strategy<Value = ConfigPayload> {
    (
        proptest::option::of(arb_string()),
        proptest::collection::hash_map(arb_nonempty_string(), arb_config_value(), 0..=MAX_ITEMS),
    )
        .prop_map(|(session_id, config)| ConfigPayload { session_id, config })
}

fn arb_gradient_batch() -> impl Strategy<Value = GradientBatch> {
    (
        arb_string(),
        any::<u64>(),
        proptest::collection::vec(
            (
                arb_string(),
                proptest::collection::vec(arb_f32(), 0..=MAX_ITEMS),
                proptest::option::of(proptest::collection::vec(arb_f32(), 0..=MAX_ITEMS)),
            ),
            0..=4,
        ),
    )
        .prop_map(|(session_id, batch_id, rows)| GradientBatch {
            session_id,
            batch_id,
            gradients: rows
                .into_iter()
                .enumerate()
                // `TraceBatch`/`GradientBatch` validate unique ids; suffix each
                // generated `layer_id` with its index to guarantee uniqueness.
                .map(
                    |(index, (layer_id, gradients, eligibility_trace))| GradientUpdate {
                        layer_id: format!("{layer_id}#{index}"),
                        gradients,
                        eligibility_trace,
                    },
                )
                .collect(),
        })
}

fn arb_trace_batch() -> impl Strategy<Value = TraceBatch> {
    (
        arb_string(),
        any::<u64>(),
        proptest::collection::vec((any::<u16>(), arb_f32(), any::<u32>()), 0..=4),
    )
        .prop_map(|(session_id, batch_id, rows)| TraceBatch {
            session_id,
            batch_id,
            traces: rows
                .into_iter()
                .enumerate()
                // Unique `channel_id` required by `TraceBatch::validate`.
                .map(
                    |(index, (channel_id, trace_value, last_spike_time))| TraceData {
                        channel_id: channel_id.wrapping_add(index as u16),
                        trace_value,
                        last_spike_time,
                    },
                )
                .collect(),
        })
}

/// Any `IpcMessage` variant within the losslessly representable domain
/// (`ConfigUpdate` uses the no-`Integer` `ConfigValue` generator).
fn arb_ipc_message() -> impl Strategy<Value = IpcMessage> {
    prop_oneof![
        arb_spike_batch().prop_map(IpcMessage::Spikes),
        arb_embedding_batch().prop_map(IpcMessage::Embeddings),
        arb_stimulus_batch().prop_map(IpcMessage::Stimuli),
        arb_neuromodulator_snapshot().prop_map(IpcMessage::Neuromodulators),
        arb_f32().prop_map(IpcMessage::Loss),
        arb_config_payload().prop_map(IpcMessage::ConfigUpdate),
        arb_gradient_batch().prop_map(IpcMessage::GradientUpdate),
        arb_trace_batch().prop_map(IpcMessage::EligibilityTraces),
        Just(IpcMessage::TrainingComplete),
        Just(IpcMessage::Shutdown),
        Just(IpcMessage::Ping),
    ]
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 64,
        // Default `failure_persistence` already records minimized failures to
        // `tests/wire_properties.proptest-regressions/` for replay.
        ..ProptestConfig::default()
    }
}

/// Decode helper: the payload object of an encoded canonical envelope.
fn payload_value(bytes: &[u8]) -> Value {
    let Value::Object(envelope) = serde_json::from_slice(bytes).unwrap() else {
        panic!("canonical encoding must be a JSON object")
    };
    envelope.get("payload").cloned().unwrap()
}

fn encode_with_payload(payload: Value) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "wire_version": 1_u32,
        "payload": payload,
    }))
    .unwrap()
}

proptest! {
    #![proptest_config(config())]

    /// Canonical encode -> public byte decode is exact identity over the
    /// losslessly representable domain (every current variant).
    #[test]
    fn canonical_round_trip_is_identity(msg in arb_ipc_message()) {
        let bytes = encode_canonical_ipc_message(&msg).unwrap();
        prop_assert_eq!(decode_ipc_message_json(&bytes).unwrap(), msg);
    }

    /// `ConfigValue::Integer` normalizes to `Float` through JSON; integers
    /// inside the f32 exact-integer range (|i| <= 2^24) round-trip to the
    /// numerically identical float.
    #[test]
    fn config_integer_normalizes_to_float_below_2pow24(i in 0_u64..=(1 << 24)) {
        let mut config = HashMap::new();
        config.insert("k".to_string(), ConfigValue::Integer(i));
        let msg = IpcMessage::ConfigUpdate(ConfigPayload {
            session_id: None,
            config,
        });
        let bytes = encode_canonical_ipc_message(&msg).unwrap();
        let decoded = decode_ipc_message_json(&bytes).unwrap();
        let IpcMessage::ConfigUpdate(payload) = decoded else {
            return Err(TestCaseError::fail("expected ConfigUpdate"));
        };
        prop_assert_eq!(
            payload.config.get("k"),
            Some(&ConfigValue::Float(i as f32))
        );
    }

    /// Unknown fields on the envelope object and inside struct payloads are
    /// ignored (forward-compatibility rule).
    #[test]
    fn unknown_envelope_and_struct_fields_are_ignored(msg in arb_ipc_message()) {
        let bytes = encode_canonical_ipc_message(&msg).unwrap();
        let mut payload = payload_value(&bytes);

        // Inject an unknown field into the first struct-level object we find.
        let target = match &mut payload {
            Value::Object(map) => map.values_mut().next(),
            _ => None,
        };
        if let Some(Value::Object(inner)) = target {
            inner.insert("unknown_field".to_string(), serde_json::json!({"nested": 1}));
        }
        let wire = encode_with_payload(payload);
        prop_assert_eq!(decode_ipc_message_json(&wire).unwrap(), msg.clone());

        // Unknown top-level envelope key is also ignored.
        let Value::Object(mut envelope) = serde_json::from_slice(&bytes).unwrap() else {
            unreachable!()
        };
        envelope.insert("future_field".to_string(), serde_json::json!(true));
        let wire = serde_json::to_vec(&envelope).unwrap();
        prop_assert_eq!(decode_ipc_message_json(&wire).unwrap(), msg);
    }

    /// Unknown variant names never become a valid message, in either the
    /// legacy tagged shape or a supported envelope.
    #[test]
    fn unknown_variants_are_rejected(name in "[A-Z][a-zA-Z]{4,16}") {
        const KNOWN: &[&str] = &[
            "Spikes", "Embeddings", "Stimuli", "Neuromodulators", "Loss",
            "ConfigUpdate", "GradientUpdate", "EligibilityTraces",
            "TrainingComplete", "Shutdown", "Ping",
        ];
        prop_assume!(!KNOWN.contains(&name.as_str()));

        let legacy = serde_json::to_vec(&serde_json::json!({ name.clone(): {} })).unwrap();
        prop_assert!(decode_ipc_message_json(&legacy).is_err());

        let enveloped = encode_with_payload(serde_json::json!({ name: {} }));
        prop_assert!(decode_ipc_message_json(&enveloped).is_err());
    }

    /// The `neuron_id` wire alias decodes into the canonical `channel_id`
    /// field, and canonical encoding emits `channel_id` only.
    #[test]
    fn neuron_id_alias_decodes_to_channel_id(
        channel_id in any::<u16>(),
        trace_value in arb_f32(),
        last_spike_time in any::<u32>(),
    ) {
        let payload = serde_json::json!({
            "EligibilityTraces": {
                "session_id": "s",
                "batch_id": 1_u64,
                "traces": [{
                    "neuron_id": channel_id,
                    "trace_value": trace_value,
                    "last_spike_time": last_spike_time,
                }]
            }
        });
        let wire = encode_with_payload(payload);
        let decoded = decode_ipc_message_json(&wire).unwrap();
        let IpcMessage::EligibilityTraces(batch) = decoded else {
            return Err(TestCaseError::fail("expected EligibilityTraces"));
        };
        prop_assert_eq!(batch.traces[0].channel_id, channel_id);

        // Canonical re-encoding surfaces `channel_id`, never the alias.
        let canonical = encode_canonical_ipc_message(&IpcMessage::EligibilityTraces(batch)).unwrap();
        let canonical_str = String::from_utf8(canonical).unwrap();
        prop_assert!(canonical_str.contains("\"channel_id\""));
        prop_assert!(!canonical_str.contains("neuron_id"));
    }

    /// Byte-level duplicate keys are rejected for known envelope and named
    /// struct fields (alias + canonical name count as one field).
    #[test]
    fn duplicate_known_fields_are_rejected(batch_id in any::<u64>()) {
        // Duplicate `batch_id` inside a struct payload.
        let wire = format!(
            r#"{{"wire_version":1,"payload":{{"Spikes":{{"session_id":null,"batch_id":{batch_id},"batch_id":{batch_id},"timestamp":0,"spikes":[],"metadata":null}}}}}}"#
        );
        prop_assert!(decode_ipc_message_json(wire.as_bytes()).is_err());

        // `neuron_id` and `channel_id` in the same object are one duplicated
        // logical field.
        let wire = format!(
            r#"{{"wire_version":1,"payload":{{"EligibilityTraces":{{"session_id":"s","batch_id":{batch_id},"traces":[{{"channel_id":1,"neuron_id":2,"trace_value":0.0,"last_spike_time":0}}]}}}}}}"#
        );
        prop_assert!(decode_ipc_message_json(wire.as_bytes()).is_err());
    }

    /// Integer-only JSON fields reject floating-point syntax at the byte
    /// decode entry point.
    #[test]
    fn integer_fields_reject_float_syntax(value in 0.5_f64..1.0e6) {
        let wire = format!(
            r#"{{"wire_version":1,"payload":{{"Spikes":{{"session_id":null,"batch_id":{value},"timestamp":0,"spikes":[],"metadata":null}}}}}}"#
        );
        prop_assert!(decode_ipc_message_json(wire.as_bytes()).is_err());
    }
}

/// Directly constructed non-finite `f32` inputs are rejected by the canonical
/// encoder's pre-serialization validation — one case per float-bearing field.
#[test]
fn canonical_encoder_rejects_non_finite_inputs() {
    let cases: Vec<IpcMessage> = vec![
        IpcMessage::Loss(f32::NAN),
        IpcMessage::Loss(f32::INFINITY),
        IpcMessage::Loss(f32::NEG_INFINITY),
        IpcMessage::Spikes(SpikeBatch {
            session_id: None,
            batch_id: 0,
            timestamp: 0,
            spikes: vec![SpikeEvent {
                channel: 0,
                time: 0,
                strength: f32::NAN,
            }],
            metadata: None,
        }),
        IpcMessage::Embeddings(EmbeddingBatch {
            session_id: None,
            batch_id: 0,
            embedding: vec![0.0, f32::INFINITY],
            sequence_length: 0,
        }),
        IpcMessage::Stimuli(StimulusBatch {
            session_id: None,
            batch_id: 0,
            timestamp: 0,
            values: vec![f32::NEG_INFINITY],
            valid_mask: None,
            metadata: None,
        }),
        IpcMessage::Neuromodulators(NeuromodulatorSnapshot {
            tick: 0,
            dopamine: f32::NAN,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 1.0,
        }),
        IpcMessage::ConfigUpdate(ConfigPayload {
            session_id: None,
            config: HashMap::from([("k".to_string(), ConfigValue::Float(f32::NAN))]),
        }),
        IpcMessage::GradientUpdate(GradientBatch {
            session_id: "s".to_string(),
            batch_id: 0,
            gradients: vec![GradientUpdate {
                layer_id: "l".to_string(),
                gradients: vec![f32::INFINITY],
                eligibility_trace: None,
            }],
        }),
        IpcMessage::EligibilityTraces(TraceBatch {
            session_id: "s".to_string(),
            batch_id: 0,
            traces: vec![TraceData {
                channel_id: 0,
                trace_value: f32::NAN,
                last_spike_time: 0,
            }],
        }),
    ];
    for msg in cases {
        assert!(
            encode_canonical_ipc_message(&msg).is_err(),
            "non-finite input must fail canonical validation: {msg:?}"
        );
    }
}

/// Above the f32 exact-integer boundary, `Integer` still normalizes to `Float`
/// (the value may lose precision — that is the documented schema, and the test
/// locks in normalization rather than impossible universal equality).
#[test]
fn config_integer_above_2pow24_still_normalizes_to_float() {
    let mut config = HashMap::new();
    config.insert("k".to_string(), ConfigValue::Integer((1 << 24) + 1));
    let msg = IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: None,
        config,
    });
    let bytes = encode_canonical_ipc_message(&msg).unwrap();
    let decoded = decode_ipc_message_json(&bytes).unwrap();
    let IpcMessage::ConfigUpdate(payload) = decoded else {
        panic!("expected ConfigUpdate");
    };
    assert_eq!(
        payload.config.get("k"),
        Some(&ConfigValue::Float(((1 << 24) + 1) as f32))
    );
}

/// A `serde_json::Value` has already lost duplicate-key information, so the
/// `Value` decode entry point cannot reject duplicates; document that the
/// byte-level path is the duplicate-rejecting surface.
#[test]
fn parsed_value_entry_point_documents_duplicate_limitation() {
    let value: Value = serde_json::from_str(
        r#"{"wire_version":1,"payload":{"Spikes":{"session_id":null,"batch_id":1,"timestamp":0,"spikes":[],"metadata":null}}}"#,
    )
    .unwrap();
    assert!(corpus_ipc::decode_ipc_message_value(value).is_ok());
}
