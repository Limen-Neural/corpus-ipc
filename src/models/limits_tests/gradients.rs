// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn max_gradient_batch(limits: ProtocolLimits) -> GradientBatch {
    GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: (0..limits.max_gradients)
            .map(|i| GradientUpdate {
                layer_id: format!("l{i}"),
                gradients: vec![0.0],
                eligibility_trace: None,
            })
            .collect(),
    }
}

#[test]
fn gradients_accept_exact_max() {
    let limits = limits();
    // 65536 rows + 65536 inner values = 131072, which is exact aggregate max.
    max_gradient_batch(limits)
        .validate_with(limits)
        .expect("gradient max");
}

#[test]
fn gradients_reject_max_plus_one() {
    let limits = limits();
    let mut gradients = max_gradient_batch(limits);
    gradients.gradients.push(GradientUpdate {
        layer_id: "extra".into(),
        gradients: vec![0.0],
        eligibility_trace: None,
    });
    let err = gradients.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
}

#[test]
fn duplicate_layer_id_is_rejected() {
    let gradients = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: vec![
            GradientUpdate {
                layer_id: "shared".into(),
                gradients: vec![0.0],
                eligibility_trace: None,
            },
            GradientUpdate {
                layer_id: "shared".into(),
                gradients: vec![1.0],
                eligibility_trace: None,
            },
        ],
    };
    let err = gradients.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::DuplicateIdentifier);
}

#[test]
fn gradient_update_round_trips() {
    let update = GradientUpdate {
        layer_id: "l0".into(),
        gradients: vec![0.25],
        eligibility_trace: Some(vec![0.1]),
    };
    let json = serde_json::to_value(&update).unwrap();
    let decoded: GradientUpdate = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.eligibility_trace.as_deref(), Some(&[0.1][..]));
}

#[test]
fn gradient_batch_round_trips_and_dispatches() {
    let update = sample_gradient_update();
    let batch = GradientBatch {
        session_id: "s".into(),
        batch_id: 9,
        gradients: vec![update],
    };
    let json = serde_json::to_value(&batch).unwrap();
    let decoded: GradientBatch = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.gradients.len(), 1);
    IpcMessage::GradientUpdate(batch).validate().unwrap();
}
