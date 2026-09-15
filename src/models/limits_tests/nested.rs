// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn parent_stimulus_rejects_nested_empty_key() {
    let parent = StimulusBatch {
        metadata: Some({
            let mut meta = BatchMetadata::default();
            meta.custom.insert(String::new(), "v".into());
            meta
        }),
        ..Default::default()
    };
    assert_eq!(
        parent.validate().unwrap_err().kind,
        ValidationKind::NestedMetadata
    );
}

#[test]
fn stimulus_with_valid_metadata_validates() {
    let ok_stim = StimulusBatch {
        values: vec![0.0],
        metadata: Some(BatchMetadata {
            processing_latency_ns: None,
            source: None,
            custom: std::collections::HashMap::from([("a".into(), "b".into())]),
        }),
        ..Default::default()
    };
    ok_stim.validate().unwrap();
    IpcMessage::Stimuli(ok_stim).validate().unwrap();
}

#[test]
fn stimulus_parent_rejects_empty_key_after_round_trip_setup() {
    let mut metadata = BatchMetadata {
        processing_latency_ns: Some(3),
        source: Some("encoder".into()),
        custom: std::collections::HashMap::from([("k".into(), "v".into())]),
    };
    metadata.custom.insert(String::new(), "v".into());
    let parent = StimulusBatch {
        values: vec![0.0],
        metadata: Some(metadata),
        ..Default::default()
    };
    assert_eq!(
        parent.validate().unwrap_err().kind,
        ValidationKind::NestedMetadata
    );
}
