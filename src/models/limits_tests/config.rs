// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn duplicate_config_keys_are_rejected() {
    let raw = r#"{"session_id":null,"config":{"a":1.0,"a":2.0}}"#;
    let err = serde_json::from_str::<ConfigPayload>(raw)
        .expect_err("duplicate keys must be rejected")
        .to_string();
    assert!(
        err.contains(ValidationKind::DuplicateIdentifier.as_str()),
        "duplicate key error: {err}"
    );
}

#[test]
fn config_map_accepts_exact_max_and_rejects_overflow() {
    let mut payload = ConfigPayload {
        session_id: None,
        config: std::collections::HashMap::new(),
    };
    for i in 0..limits().max_metadata_entries {
        payload
            .config
            .insert(format!("k{i}"), ConfigValue::Boolean(true));
    }
    payload.validate().unwrap();
    payload
        .config
        .insert("overflow".into(), ConfigValue::Boolean(false));
    assert_eq!(
        payload.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
}

#[test]
fn serde_rejects_oversize_config_map() {
    let mut oversize = serde_json::Map::new();
    for i in 0..=limits().max_metadata_entries {
        oversize.insert(format!("k{i}"), serde_json::json!(true));
    }
    assert_serde_kind::<ConfigPayload>(
        serde_json::json!({"session_id": null, "config": oversize}),
        ValidationKind::LimitExceeded,
    );
}

#[test]
fn config_payload_round_trips() {
    let mut config = std::collections::HashMap::new();
    config.insert("mode".into(), ConfigValue::String("fast".into()));
    config.insert("n".into(), ConfigValue::Integer(7));
    config.insert("on".into(), ConfigValue::Boolean(true));
    config.insert("arr".into(), ConfigValue::FloatArray(vec![1.0, 2.0]));
    let payload = ConfigPayload {
        session_id: None,
        config,
    };
    payload.validate().unwrap();
    let json = serde_json::to_value(&payload).unwrap();
    let decoded: ConfigPayload = serde_json::from_value(json).unwrap();
    assert!(matches!(
        decoded.config.get("on"),
        Some(ConfigValue::Boolean(true))
    ));
    IpcMessage::ConfigUpdate(payload).validate().unwrap();
    ConfigValue::Integer(3).validate().unwrap();
    ConfigValue::Boolean(false).validate().unwrap();
}
