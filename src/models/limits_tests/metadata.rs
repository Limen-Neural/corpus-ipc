// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn filled_metadata(limits: ProtocolLimits) -> BatchMetadata {
    let mut metadata = BatchMetadata::default();
    for i in 0..limits.max_metadata_entries {
        metadata.custom.insert(format!("k{i}"), "v".into());
    }
    metadata
}

#[test]
fn metadata_accepts_exact_max_entries() {
    let limits = limits();
    filled_metadata(limits)
        .validate_with(limits)
        .expect("metadata max");
}

#[test]
fn metadata_rejects_max_plus_one_entry() {
    let limits = limits();
    let mut metadata = filled_metadata(limits);
    metadata.custom.insert("overflow".into(), "v".into());
    let err = metadata.validate_with(limits).unwrap_err();
    assert_limit_exceeded(&err, "custom");
}

#[test]
fn empty_metadata_key_is_nested_metadata() {
    let mut nested = BatchMetadata::default();
    nested.custom.insert(String::new(), "v".into());
    let err = nested.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::NestedMetadata);
}

#[test]
fn nested_object_in_custom_fails_to_deserialize() {
    let nested_object = serde_json::json!({
        "processing_latency_ns": null,
        "source": null,
        "custom": {"k": {"nested": true}}
    });
    assert!(
        serde_json::from_value::<BatchMetadata>(nested_object).is_err(),
        "nested object in metadata.custom must fail to deserialize"
    );
}

#[test]
fn metadata_round_trips() {
    let metadata = BatchMetadata {
        processing_latency_ns: Some(3),
        source: Some("encoder".into()),
        custom: std::collections::HashMap::from([("k".into(), "v".into())]),
    };
    let json = serde_json::to_value(&metadata).unwrap();
    let decoded: BatchMetadata = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.source.as_deref(), Some("encoder"));
}

#[test]
fn spike_batch_with_metadata_validates() {
    let metadata = BatchMetadata {
        processing_latency_ns: Some(3),
        source: Some("encoder".into()),
        custom: std::collections::HashMap::from([("k".into(), "v".into())]),
    };
    let spikes = SpikeBatch {
        session_id: Some("sess".into()),
        batch_id: 1,
        timestamp: 0,
        spikes: vec![SpikeEvent {
            channel: 1,
            time: 2,
            strength: 0.3,
        }],
        metadata: Some(metadata),
    };
    spikes.validate().unwrap();
    IpcMessage::Spikes(spikes).validate().unwrap();
}
