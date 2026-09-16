// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn session_id_accepts_exact_max_bytes() {
    let limits = limits();
    let session = SpikeBatch {
        session_id: Some(max_string(limits.max_string_bytes)),
        ..Default::default()
    };
    session.validate_with(limits).expect("string max");
}

#[test]
fn session_id_rejects_max_plus_one_bytes() {
    let limits = limits();
    let oversize = SpikeBatch {
        session_id: Some(oversize_string(limits.max_string_bytes)),
        ..Default::default()
    };
    let err = oversize.validate_with(limits).unwrap_err();
    assert_limit_exceeded(&err, "session_id");
    assert_eq!(
        err.actual,
        Some(ValidationMeasure::Bytes(
            u64::try_from(limits.max_string_bytes + 1).unwrap()
        ))
    );
}

#[test]
fn oversize_spike_session_id_is_rejected() {
    let too_long = oversize_string(limits().max_string_bytes);
    let err = SpikeBatch {
        session_id: Some(too_long),
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "session_id");
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
}

#[test]
fn oversize_trace_session_id_is_rejected() {
    let too_long = oversize_string(limits().max_string_bytes);
    let err = TraceBatch {
        session_id: too_long,
        batch_id: 1,
        traces: vec![],
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "session_id");
}

#[test]
fn oversize_metadata_key_value_and_source_are_rejected() {
    let too_long = oversize_string(limits().max_string_bytes);
    let mut metadata = BatchMetadata::default();
    metadata.custom.insert(too_long.clone(), "v".into());
    assert_eq!(
        metadata.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
    metadata.custom.clear();
    metadata.custom.insert("k".into(), too_long.clone());
    assert_eq!(
        metadata.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
    metadata.custom.clear();
    metadata.source = Some(too_long);
    assert_eq!(metadata.validate().unwrap_err().path, "source");
}

#[test]
fn oversize_layer_id_is_rejected() {
    let err = GradientUpdate {
        layer_id: oversize_string(limits().max_string_bytes),
        gradients: vec![0.0],
        eligibility_trace: None,
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "layer_id");
}
