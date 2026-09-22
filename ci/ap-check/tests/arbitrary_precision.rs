// SPDX-License-Identifier: MIT OR Apache-2.0

//! serde_json `arbitrary_precision` feature-unification check for corpus-ipc.
//!
//! This package enables `serde_json/arbitrary_precision` and depends on
//! `corpus-ipc` via path, so Cargo feature unification turns
//! `arbitrary_precision` on for `corpus-ipc` in this build. That reproduces
//! exactly what a consumer hits when it unifies the feature onto the graph.
//! The corpus-ipc crate itself no longer publishes any feature to simulate
//! this; the check lives here so `--all-features` on the published crate does
//! not flip serde_json semantics for consumers.
//!
//! These tests exercise only corpus-ipc's public API.

use std::collections::HashMap;

use corpus_ipc::{
    ConfigPayload, ConfigValue, IpcMessage, decode_ipc_message_json, encode_canonical_ipc_message,
};

/// Canonical encoding must stay f64-promoted even under `arbitrary_precision`
/// unification. Default serde_json stores `f32` as `f as f64`, so `0.1_f32` is
/// `0.10000000149011612` on the wire. With `arbitrary_precision` enabled a
/// naive path could emit shortest-f32 `0.1` instead; this byte-exact lock
/// proves the contract holds. Mirrors tests/canonical.rs
/// `canonical_f32_bytes_match_f64_promotion_not_shortest_f32`.
#[test]
fn canonical_f32_bytes_match_f64_promotion_not_shortest_f32_under_ap() {
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

/// ConfigValue JSON decimals must still decode under `arbitrary_precision`
/// unification, round-tripping through the public encode/decode API.
#[test]
fn config_value_decimals_decode_under_ap() {
    let mut config = HashMap::new();
    config.insert("a".to_string(), ConfigValue::Float(1.0));
    config.insert("b".to_string(), ConfigValue::Float(0.1));
    config.insert("arr".to_string(), ConfigValue::FloatArray(vec![1.0, 2.0]));
    let msg = IpcMessage::ConfigUpdate(ConfigPayload {
        session_id: Some("sess-1".into()),
        config,
    });

    let bytes = encode_canonical_ipc_message(&msg).unwrap();
    let decoded = decode_ipc_message_json(&bytes).unwrap();
    let IpcMessage::ConfigUpdate(payload) = decoded else {
        panic!("expected ConfigUpdate");
    };
    assert_eq!(payload.config.get("a"), Some(&ConfigValue::Float(1.0)));
    assert_eq!(payload.config.get("b"), Some(&ConfigValue::Float(0.1)));
    assert_eq!(
        payload.config.get("arr"),
        Some(&ConfigValue::FloatArray(vec![1.0, 2.0]))
    );
}

/// Item 3 (re-homed from src/models/limits_tests/config.rs): under
/// `arbitrary_precision` unification, JSON decimals reach `deserialize_any` as
/// the synthetic number map handled by `visit_map`. The decimal token must be
/// parsed straight to `f32` rather than via an intermediate `f64` widening,
/// otherwise double rounding can pick the wrong neighbouring `f32`. This token
/// sits just above an `f32` midpoint where the two paths disagree: direct
/// parse -> `0x3f000003`, via-`f64` -> `0x3f000002`.
#[test]
fn config_value_visitor_parses_decimal_directly_to_f32_under_ap() {
    const TOKEN: &str = "0.5000001490116119384765625000000000000000000000000000000000001";
    let direct = TOKEN.parse::<f32>().expect("token parses as f32");
    let via_f64 = TOKEN.parse::<f64>().expect("token parses as f64") as f32;
    assert_eq!(direct.to_bits(), 0x3f00_0003);
    assert_eq!(via_f64.to_bits(), 0x3f00_0002);
    assert_ne!(
        direct.to_bits(),
        via_f64.to_bits(),
        "token must expose the double-rounding difference"
    );

    let raw = format!(r#"{{"session_id":null,"config":{{"a":{TOKEN}}}}}"#);
    let payload = serde_json::from_str::<ConfigPayload>(&raw).expect("AP decimal must decode");
    assert_eq!(
        payload.config.get("a"),
        Some(&ConfigValue::Float(direct)),
        "decimal must be parsed directly to f32, not via f64 double rounding"
    );
}
