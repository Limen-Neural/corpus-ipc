// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical wire-v1 encoding tests (LIM-1326).
//!
//! Covers the project wire profile documented in `docs/wire-encoding.md`:
//! envelope/unit forms, sorted nested maps, Unicode, signed zero, the
//! documented `Integer(42)` -> `Float(42.0)` + precision-loss normalization,
//! and non-finite rejection before serialization. In-process only.

use std::collections::HashMap;

use corpus_ipc::{
    BatchMetadata, CanonicalEncodeError, ConfigPayload, ConfigValue, IpcMessage, SpikeBatch,
    SpikeEvent, decode_ipc_message_json, encode_canonical_ipc_message,
};

fn config_message(config: HashMap<String, ConfigValue>) -> IpcMessage {
    IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: Some("sess-1".into()),
        config,
    })
}

#[test]
fn canonical_envelope_form_round_trips() {
    let msg = IpcMessage::Spikes(SpikeBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        spikes: vec![SpikeEvent {
            channel: 3,
            time: 11,
            strength: 0.5,
        }],
        metadata: None,
    });
    let bytes = encode_canonical_ipc_message(&msg).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    // Canonical form sorts keys at every level, including the envelope, so
    // `payload` sorts before `wire_version`.
    assert!(
        text.starts_with(r#"{"payload":{"Spikes":"#),
        "envelope payload must come first under sorted keys: {text}"
    );
    assert!(
        text.ends_with(r#""wire_version":1}"#),
        "wire_version must sort last in the envelope: {text}"
    );
    assert_eq!(decode_ipc_message_json(&bytes).unwrap(), msg);
}

#[test]
fn canonical_unit_variant_encodes_as_string_payload() {
    let bytes = encode_canonical_ipc_message(&IpcMessage::Ping).unwrap();
    // Sorted envelope keys: `payload` before `wire_version`.
    assert_eq!(bytes, br#"{"payload":"Ping","wire_version":1}"#);
    assert_eq!(decode_ipc_message_json(&bytes).unwrap(), IpcMessage::Ping);
}

#[test]
fn canonical_nested_map_keys_are_sorted_regardless_of_insertion_order() {
    // Insert in two different orders; both must produce byte-identical output
    // with keys sorted ascending.
    let mut a = HashMap::new();
    a.insert("zeta".to_string(), ConfigValue::Boolean(true));
    a.insert("alpha".to_string(), ConfigValue::Boolean(false));
    a.insert("mu".to_string(), ConfigValue::Boolean(true));

    let mut b = HashMap::new();
    b.insert("mu".to_string(), ConfigValue::Boolean(true));
    b.insert("alpha".to_string(), ConfigValue::Boolean(false));
    b.insert("zeta".to_string(), ConfigValue::Boolean(true));

    let bytes_a = encode_canonical_ipc_message(&config_message(a)).unwrap();
    let bytes_b = encode_canonical_ipc_message(&config_message(b)).unwrap();
    assert_eq!(bytes_a, bytes_b, "insertion order must not affect bytes");

    let text = String::from_utf8(bytes_a).unwrap();
    let alpha = text.find("alpha").unwrap();
    let mu = text.find("mu").unwrap();
    let zeta = text.find("zeta").unwrap();
    assert!(alpha < mu && mu < zeta, "keys must be sorted: {text}");
}

#[test]
fn canonical_repeated_encoding_is_stable_across_calls() {
    let mut config = HashMap::new();
    for k in ["k3", "k1", "k2", "k0"] {
        config.insert(k.to_string(), ConfigValue::Float(1.0));
    }
    let msg = config_message(config);
    let first = encode_canonical_ipc_message(&msg).unwrap();
    for _ in 0..64 {
        assert_eq!(encode_canonical_ipc_message(&msg).unwrap(), first);
    }
}

#[test]
fn canonical_preserves_unicode_keys_and_values() {
    let mut config = HashMap::new();
    config.insert("café".to_string(), ConfigValue::String("naïve-Ω".into()));
    config.insert("日本語".to_string(), ConfigValue::Boolean(true));
    let bytes = encode_canonical_ipc_message(&config_message(config.clone())).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(
        text.contains("café") && text.contains("naïve-Ω") && text.contains("日本語"),
        "{text}"
    );
    // Round-trips back to the same message.
    let decoded = decode_ipc_message_json(&bytes).unwrap();
    assert_eq!(decoded, config_message(config));
}

#[test]
fn canonical_f32_bytes_match_f64_promotion_not_shortest_f32() {
    // Default serde_json Number stores f32 as `f as f64`, so 0.1_f32 is
    // 0.10000000149011612 on the wire. serde_json `arbitrary_precision` would
    // otherwise emit shortest-f32 `0.1`. This lock is the contract; re-run
    // with `--features _ci_serde_json_arbitrary_precision` (CI does) so
    // feature unification cannot silently change the bytes.
    let bytes = encode_canonical_ipc_message(&IpcMessage::Loss(0.1)).unwrap();
    assert_eq!(
        bytes,
        br#"{"payload":{"Loss":0.10000000149011612},"wire_version":1}"#
    );
    assert_eq!(
        decode_ipc_message_json(&bytes).unwrap(),
        IpcMessage::Loss(0.1)
    );
}

#[test]
fn canonical_preserves_signed_zero() {
    let mut neg = HashMap::new();
    neg.insert("z".to_string(), ConfigValue::Float(-0.0));
    let mut pos = HashMap::new();
    pos.insert("z".to_string(), ConfigValue::Float(0.0));

    let neg_bytes = encode_canonical_ipc_message(&config_message(neg)).unwrap();
    let pos_bytes = encode_canonical_ipc_message(&config_message(pos)).unwrap();
    assert_ne!(
        neg_bytes, pos_bytes,
        "signed zero must be distinct on the wire"
    );
    assert!(String::from_utf8(neg_bytes).unwrap().contains("-0.0"));
}

#[test]
fn canonical_integer_normalizes_to_float_and_documents_precision_loss() {
    // Documented normalization: Integer(42) round-trips as Float(42.0).
    let mut config = HashMap::new();
    config.insert("n".to_string(), ConfigValue::Integer(42));
    let bytes = encode_canonical_ipc_message(&config_message(config)).unwrap();
    if let IpcMessage::ConfigUpdate(payload) = decode_ipc_message_json(&bytes).unwrap() {
        assert_eq!(payload.config.get("n"), Some(&ConfigValue::Float(42.0)));
    } else {
        panic!("expected ConfigUpdate");
    }

    // 16_777_217 = 2^24 + 1 is the smallest integer f32 cannot represent
    // exactly, so it loses identity on an f32 round-trip (documented).
    let big = (1u64 << 24) + 1;
    let mut config = HashMap::new();
    config.insert("big".to_string(), ConfigValue::Integer(big));
    let bytes = encode_canonical_ipc_message(&config_message(config)).unwrap();
    if let IpcMessage::ConfigUpdate(payload) = decode_ipc_message_json(&bytes).unwrap() {
        let ConfigValue::Float(v) = payload.config.get("big").unwrap() else {
            panic!("expected Float after round-trip");
        };
        assert_eq!(*v, big as f32);
        assert_ne!(*v as u64, big, "precision loss above 2^24 is expected");
    } else {
        panic!("expected ConfigUpdate");
    }
}

#[test]
fn canonical_rejects_non_finite_before_serialization() {
    let msg = IpcMessage::Loss(f32::NAN);
    let err = encode_canonical_ipc_message(&msg).expect_err("NaN must be rejected");
    assert!(
        matches!(err, CanonicalEncodeError::Validation(_)),
        "got {err}"
    );

    let mut config = HashMap::new();
    config.insert("bad".to_string(), ConfigValue::Float(f32::INFINITY));
    let err =
        encode_canonical_ipc_message(&config_message(config)).expect_err("inf must be rejected");
    assert!(
        matches!(err, CanonicalEncodeError::Validation(_)),
        "got {err}"
    );
}

#[test]
fn canonical_sorts_metadata_custom_map() {
    let mut custom = HashMap::new();
    custom.insert("b".to_string(), "1".to_string());
    custom.insert("a".to_string(), "2".to_string());
    let msg = IpcMessage::Spikes(SpikeBatch {
        session_id: None,
        batch_id: 0,
        timestamp: 0,
        spikes: vec![],
        metadata: Some(BatchMetadata {
            processing_latency_ns: None,
            source: None,
            custom,
        }),
    });
    let bytes = encode_canonical_ipc_message(&msg).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.find(r#""a""#).unwrap() < text.find(r#""b""#).unwrap(),
        "{text}"
    );
}
