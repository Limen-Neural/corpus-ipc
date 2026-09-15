// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture-backed compatibility envelope tests (RM-1333).
//!
//! These local fixtures stand in for the canonical `test-vectors/` corpus
//! (RM-1330) until that lands. They cover the unversioned-as-v1 encoding and
//! the min / current / min-1 / current+1 envelope boundaries.

use corpus_ipc::{
    CompatibilityError, EnvelopeError, IpcMessage, SpikeBatch, SpikeEvent, WireCompatibility,
    decode_ipc_message_json,
};

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
