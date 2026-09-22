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

/// Value-type preservation regression: `ConfigValue::visit_map` now
/// deserializes the incoming map to a `serde_json::Value` and accepts only
/// `Value::Number`. Under `arbitrary_precision` unification the `visit_map`
/// arm is the path decimals travel, so this asserts a genuine decimal still
/// decodes to the correct `ConfigValue::Float`. If the classification arm were
/// reverted to blindly accept any map (or if `Value::Number` were rejected)
/// this positive case would fail.
#[test]
fn config_value_number_map_still_decodes_under_ap() {
    assert_eq!(
        serde_json::from_str::<ConfigValue>("1.25").expect("AP decimal decodes"),
        ConfigValue::Float(1.25),
    );

    let raw = r#"{"session_id":null,"config":{"a":1.25,"b":0.1}}"#;
    let payload = serde_json::from_str::<ConfigPayload>(raw).expect("AP payload decodes");
    assert_eq!(payload.config.get("a"), Some(&ConfigValue::Float(1.25)));
    assert_eq!(payload.config.get("b"), Some(&ConfigValue::Float(0.1)));
}

/// Value-type preservation regression: a genuine user object reaching
/// `visit_map` must be rejected, not coerced to a float. Under
/// `arbitrary_precision` a distinctly-shaped user object (a key that is *not*
/// the synthetic number token) still surfaces as `Value::Object`, so the
/// Value-first classification rejects it. This is the property the reviewer's
/// "preserve the JSON value type before converting numbers" change guards: if
/// `visit_map` fell back to `serde_json::Number::deserialize` on the raw map,
/// or accepted any map, an object could slip through as `Float`.
#[test]
fn config_value_rejects_plain_object_under_ap() {
    serde_json::from_str::<ConfigValue>(r#"{"not":"a-number"}"#)
        .expect_err("a plain user object must be rejected, not decoded as Float");

    let raw = r#"{"session_id":null,"config":{"a":{"not":"a-number"}}}"#;
    serde_json::from_str::<ConfigPayload>(raw)
        .expect_err("a nested user object must be rejected under AP");
}

/// Documented, characterization-only limitation (arbitrary_precision ONLY).
///
/// serde_json represents a genuine JSON number and a user object literally
/// keyed `$serde_json::private::Number` with a single string value as the
/// *exact same* synthetic map when `arbitrary_precision` is enabled. Its key
/// classifier collapses both into `serde_json::Value::Number`, so at the serde
/// deserialization layer the exact-form spoof is indistinguishable from a real
/// number and is accepted as `ConfigValue::Float`. Rejecting it would require
/// rejecting genuine decimals too, which is not acceptable.
///
/// This test pins that behavior so the limitation is visible and any future
/// serde_json change that makes the spoof distinguishable (which WOULD let us
/// reject it) trips this assertion and prompts a revisit. In the DEFAULT build
/// (no `arbitrary_precision`) the same spoof arrives as `Value::Object` and IS
/// rejected - see `config_value_visitor_rejects_spoofed_number_object` in
/// src/models/limits_tests/config.rs.
#[test]
fn config_value_exact_spoof_is_indistinguishable_under_ap() {
    // Confirm the premise: serde_json itself collapses the exact-form spoof
    // into a Value::Number under arbitrary_precision.
    let as_value: serde_json::Value =
        serde_json::from_str(r#"{"$serde_json::private::Number":"1.25"}"#)
            .expect("spoof parses as a JSON value");
    assert!(
        as_value.is_number() && !as_value.is_object(),
        "under arbitrary_precision the exact-form spoof is classified as a Number, \
         got {as_value:?}; if this ever becomes an Object the spoof can be rejected"
    );

    // Therefore ConfigValue also accepts it (indistinguishable from a real
    // number). This documents the residual AP-only edge case.
    assert_eq!(
        serde_json::from_str::<ConfigValue>(r#"{"$serde_json::private::Number":"1.25"}"#)
            .expect("exact-form spoof is indistinguishable from a number under AP"),
        ConfigValue::Float(1.25),
    );
}
