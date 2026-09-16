// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compatibility envelope tests (RM-1333).
//!
//! Local fixtures stand in for the canonical `test-vectors/` corpus (RM-1330)
//! until that lands. They cover the unversioned-as-v1 encoding and the min /
//! current / min-1 / current+1 envelope boundaries.

use corpus_ipc::{
    Compatibility, CompatibilityError, EnvelopeError, IpcMessage, SpikeBatch, SpikeEvent,
    WireCompatibility, WireEnvelope, accept_wire_version, classify_wire_version,
    decode_ipc_message_json, decode_ipc_message_value, encode_ipc_message_json,
};
use serde_json::Value;

fn sample_spikes() -> IpcMessage {
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
    })
}

fn envelope_json(version: u32, payload: Value) -> Value {
    serde_json::json!({
        "wire_version": version,
        "payload": payload,
    })
}

fn sample_spikes_payload() -> Value {
    serde_json::json!({
        "Spikes": {
            "session_id": "sess-1",
            "batch_id": 7,
            "timestamp": 1_700_000_000,
            "spikes": [{
                "channel": 3,
                "time": 11,
                "strength": 0.5
            }],
            "metadata": null
        }
    })
}

#[test]
fn compatibility_legacy_unversioned_fixture_decodes() {
    let bytes = include_bytes!("fixtures/compatibility/legacy_unversioned_spikes.json");
    let decoded = decode_ipc_message_json(bytes).expect("supported older fixture must decode");
    assert_eq!(decoded, sample_spikes());
}

#[test]
fn compatibility_legacy_unversioned_unit_variant_fixture_decodes() {
    let bytes = include_bytes!("fixtures/compatibility/legacy_unversioned_ping.json");
    let decoded = decode_ipc_message_json(bytes).expect("legacy unit variant must decode");
    assert_eq!(decoded, IpcMessage::Ping);
}

#[test]
fn compatibility_supported_envelope_fixture_decodes() {
    // Committed fixture is wire version 1. It must keep decoding for as long
    // as 1 stays inside [MIN_SUPPORTED, CURRENT]; widening CURRENT must not
    // require this test to change.
    let bytes = include_bytes!("fixtures/compatibility/v1_envelope_spikes.json");
    let decoded = decode_ipc_message_json(bytes).expect("v1 envelope fixture must decode");
    assert_eq!(decoded, sample_spikes());
}

#[test]
fn compatibility_min_minus_one_fixture_is_too_old() {
    let bytes = include_bytes!("fixtures/compatibility/v0_too_old_ping.json");
    let err = decode_ipc_message_json(bytes).expect_err("min-1 must fail closed");
    match err {
        EnvelopeError::Compatibility(CompatibilityError::TooOld { found, min }) => {
            assert_eq!(found, WireCompatibility::MIN_SUPPORTED - 1);
            assert_eq!(min, WireCompatibility::MIN_SUPPORTED);
        }
        other => panic!("expected TooOld before payload use, got {other}"),
    }
}

#[test]
fn compatibility_current_plus_one_fixture_is_too_new() {
    let bytes = include_bytes!("fixtures/compatibility/v2_too_new_unknown_variant.json");
    let err = decode_ipc_message_json(bytes).expect_err("current+1 must fail closed");
    match err {
        EnvelopeError::Compatibility(CompatibilityError::TooNew { found, current }) => {
            assert_eq!(found, WireCompatibility::CURRENT + 1);
            assert_eq!(current, WireCompatibility::CURRENT);
        }
        other => panic!("expected TooNew before the unknown payload variant is used, got {other}"),
    }
}

#[test]
fn compatibility_window_is_internally_consistent() {
    const {
        assert!(WireCompatibility::MIN_SUPPORTED > 0);
        assert!(WireCompatibility::CURRENT >= WireCompatibility::MIN_SUPPORTED);
        assert!(
            WireCompatibility::LEGACY_UNVERSIONED >= WireCompatibility::MIN_SUPPORTED
                && WireCompatibility::LEGACY_UNVERSIONED <= WireCompatibility::CURRENT
        );
    }
}

#[test]
fn compatibility_table_covers_min_minus_one_min_current_and_current_plus_one() {
    let min = WireCompatibility::MIN_SUPPORTED;
    let current = WireCompatibility::CURRENT;
    let rows = [
        (min - 1, Compatibility::TooOld),
        (min, Compatibility::Supported),
        (current, Compatibility::Supported),
        (current + 1, Compatibility::TooNew),
    ];
    for (version, expected) in rows {
        assert_eq!(
            classify_wire_version(version),
            expected,
            "version {version} should be {expected:?}"
        );
    }
}

#[test]
fn compatibility_accept_returns_typed_too_old_and_too_new_errors() {
    let min = WireCompatibility::MIN_SUPPORTED;
    let current = WireCompatibility::CURRENT;

    let too_old = accept_wire_version(min - 1).expect_err("min-1 must fail");
    assert_eq!(
        too_old,
        CompatibilityError::TooOld {
            found: min - 1,
            min,
        }
    );
    let too_old_text = too_old.to_string();
    assert!(
        too_old_text.contains("too old"),
        "too-old error must be actionable: {too_old_text}"
    );
    assert!(
        too_old_text.contains("upgrade the producer"),
        "too-old error must say what to do: {too_old_text}"
    );

    let too_new = accept_wire_version(current + 1).expect_err("current+1 must fail");
    assert_eq!(
        too_new,
        CompatibilityError::TooNew {
            found: current + 1,
            current,
        }
    );
    let too_new_text = too_new.to_string();
    assert!(
        too_new_text.contains("too new"),
        "too-new error must be actionable: {too_new_text}"
    );
    assert!(
        too_new_text.contains("upgrade corpus-ipc"),
        "too-new error must say what to do: {too_new_text}"
    );

    let supported = accept_wire_version(current).expect("current must be accepted");
    assert_eq!(supported.get(), current);
}

#[test]
fn compatibility_envelope_json_keys_stay_stable() {
    let env = WireEnvelope::new(IpcMessage::Ping);
    let json = serde_json::to_value(&env).unwrap();
    // Externally tagged unit variants encode as a JSON string, not
    // `{"Ping":null}`. Struct variants still wrap in an object.
    assert_eq!(
        json,
        serde_json::json!({
            "wire_version": 1,
            "payload": "Ping"
        })
    );
}

#[test]
fn compatibility_current_envelope_round_trips_before_payload_use() {
    let encoded = encode_ipc_message_json(&sample_spikes()).unwrap();
    let decoded = decode_ipc_message_json(&encoded).unwrap();
    assert_eq!(decoded, sample_spikes());
}

#[test]
fn compatibility_legacy_unversioned_value_decodes() {
    let legacy = sample_spikes_payload();
    let decoded = decode_ipc_message_value(legacy).unwrap();
    assert_eq!(decoded, sample_spikes());
}

#[test]
fn compatibility_legacy_unit_variant_string_decodes() {
    let decoded = decode_ipc_message_json(br#""Ping""#).unwrap();
    assert_eq!(decoded, IpcMessage::Ping);
    let encoded = encode_ipc_message_json(&IpcMessage::Ping).unwrap();
    assert_eq!(decode_ipc_message_json(&encoded).unwrap(), IpcMessage::Ping);
}

#[test]
fn compatibility_min_supported_envelope_decodes() {
    let json = envelope_json(WireCompatibility::MIN_SUPPORTED, sample_spikes_payload());
    let decoded = decode_ipc_message_value(json).unwrap();
    assert_eq!(decoded, sample_spikes());
}

#[test]
fn compatibility_too_old_envelope_fails_before_payload_use() {
    let json = envelope_json(
        WireCompatibility::MIN_SUPPORTED - 1,
        serde_json::json!({ "Ping": null }),
    );
    let err = decode_ipc_message_value(json).expect_err("too-old must fail");
    match err {
        EnvelopeError::Compatibility(CompatibilityError::TooOld { found, min }) => {
            assert_eq!(found, WireCompatibility::MIN_SUPPORTED - 1);
            assert_eq!(min, WireCompatibility::MIN_SUPPORTED);
        }
        other => panic!("expected TooOld, got {other}"),
    }
}

#[test]
fn compatibility_too_new_envelope_fails_before_unknown_payload_is_used() {
    // A future producer might send an unknown variant. The version check
    // must reject this as TooNew rather than turning it into Ping/Default
    // or a payload-unknown-variant error.
    let json = envelope_json(
        WireCompatibility::CURRENT + 1,
        serde_json::json!({ "BrandNewFutureMessage": { "x": 1 } }),
    );
    let err = decode_ipc_message_value(json).expect_err("too-new must fail");
    match err {
        EnvelopeError::Compatibility(CompatibilityError::TooNew { found, current }) => {
            assert_eq!(found, WireCompatibility::CURRENT + 1);
            assert_eq!(current, WireCompatibility::CURRENT);
        }
        other => panic!("expected TooNew before payload use, got {other}"),
    }
}

#[test]
fn compatibility_too_new_missing_payload_still_fails_on_version() {
    let json = serde_json::json!({ "wire_version": WireCompatibility::CURRENT + 1 });
    let err = WireEnvelope::<IpcMessage>::from_json_value(json)
        .expect_err("too-new must fail even without payload");
    assert!(matches!(
        err,
        EnvelopeError::Compatibility(CompatibilityError::TooNew { .. })
    ));
}

#[test]
fn compatibility_unknown_fields_are_ignored_on_supported_payloads() {
    let json = serde_json::json!({
        "Spikes": {
            "session_id": "sess-1",
            "batch_id": 7,
            "timestamp": 1_700_000_000,
            "spikes": [{
                "channel": 3,
                "time": 11,
                "strength": 0.5
            }],
            "metadata": null,
            "future_optional_field": 123
        }
    });
    let decoded: SpikeBatch = serde_json::from_value(json.get("Spikes").unwrap().clone()).unwrap();
    assert_eq!(decoded.batch_id, 7);
    let via_entry = decode_ipc_message_value(json).unwrap();
    assert_eq!(via_entry, sample_spikes());
}

#[test]
fn compatibility_unknown_envelope_fields_are_ignored() {
    let mut json = envelope_json(
        WireCompatibility::CURRENT,
        serde_json::json!({ "Ping": null }),
    );
    json.as_object_mut()
        .unwrap()
        .insert("future_envelope_field".into(), serde_json::json!("ok"));
    let decoded = decode_ipc_message_value(json).unwrap();
    assert_eq!(decoded, IpcMessage::Ping);
}

#[test]
fn compatibility_unknown_variant_never_becomes_a_valid_default() {
    let json = serde_json::json!({ "NotARealVariant": null });
    assert!(
        serde_json::from_value::<IpcMessage>(json.clone()).is_err(),
        "unknown IpcMessage variants must fail closed"
    );
    let err = decode_ipc_message_value(json).expect_err("unknown variant must fail");
    assert!(
        matches!(err, EnvelopeError::Payload(_)),
        "legacy unknown variant is a payload error, not a default: {err}"
    );

    let enveloped = envelope_json(
        WireCompatibility::CURRENT,
        serde_json::json!({ "NotARealVariant": null }),
    );
    let err = decode_ipc_message_value(enveloped).expect_err("unknown variant must fail");
    assert!(
        matches!(err, EnvelopeError::Payload(_)),
        "supported-version unknown variant is a payload error: {err}"
    );
}

#[test]
fn compatibility_decode_json_non_object_is_not_an_object() {
    let err = WireEnvelope::<IpcMessage>::decode_json(br#""Ping""#).unwrap_err();
    assert!(
        matches!(err, EnvelopeError::NotAnObject),
        "string JSON should be NotAnObject, got {err}"
    );
    let err = WireEnvelope::<IpcMessage>::decode_json(b"[]").unwrap_err();
    assert!(
        matches!(err, EnvelopeError::NotAnObject),
        "array JSON should be NotAnObject, got {err}"
    );
}

#[test]
fn compatibility_decode_json_malformed_is_json_error() {
    let err = WireEnvelope::<IpcMessage>::decode_json(b"{").unwrap_err();
    assert!(matches!(err, EnvelopeError::Json(_)));
}

#[test]
fn compatibility_duplicate_wire_version_is_rejected() {
    let bytes = br#"{"wire_version":1,"wire_version":2,"payload":"Ping"}"#;
    let err = decode_ipc_message_json(bytes).expect_err("duplicate wire_version must fail");
    assert!(
        matches!(err, EnvelopeError::Json(_)),
        "duplicate keys must not fall through to last-win Value decode: {err}"
    );
}

#[test]
fn compatibility_null_payload_deserializes_when_t_accepts_null() {
    let bytes = br#"{"wire_version":1,"payload":null}"#;
    let decoded = WireEnvelope::<Option<u32>>::decode_json(bytes).unwrap();
    assert_eq!(decoded.payload, None);
    let err = WireEnvelope::<Option<u32>>::decode_json(br#"{"wire_version":1}"#).unwrap_err();
    assert!(matches!(err, EnvelopeError::MissingPayload));
    let err = decode_ipc_message_json(bytes).unwrap_err();
    assert!(
        matches!(err, EnvelopeError::Payload(_)),
        "IpcMessage must not treat null payload as missing: {err}"
    );
}

#[test]
fn compatibility_null_wire_version_is_invalid_version() {
    let bytes = br#"{"wire_version":null,"payload":"Ping"}"#;
    let err = decode_ipc_message_json(bytes).unwrap_err();
    assert!(
        matches!(err, EnvelopeError::InvalidVersion(ref s) if s == "null"),
        "null wire_version from bytes should be InvalidVersion(\"null\"), got {err:?}"
    );

    let val: Value = serde_json::from_slice(bytes).unwrap();
    let err = decode_ipc_message_value(val).unwrap_err();
    assert!(
        matches!(err, EnvelopeError::InvalidVersion(ref s) if s == "null"),
        "null wire_version from Value should be InvalidVersion(\"null\"), got {err:?}"
    );

    let err = WireEnvelope::<IpcMessage>::decode_json(bytes).unwrap_err();
    assert!(
        matches!(err, EnvelopeError::InvalidVersion(ref s) if s == "null"),
        "null wire_version in WireEnvelope should be InvalidVersion(\"null\"), got {err:?}"
    );
}
