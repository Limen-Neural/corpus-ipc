// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn aggregate_accepts_exact_max() {
    let limits = limits();
    let aggregate = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: vec![GradientUpdate {
            layer_id: "a".into(),
            gradients: vec![0.0; 4],
            eligibility_trace: Some(vec![0.0; 4]),
        }],
    };
    let tight = ProtocolLimits {
        max_aggregate_records: 9,
        ..limits
    };
    aggregate.validate_with(tight).expect("aggregate max");
}

#[test]
fn aggregate_rejects_max_plus_one() {
    let limits = limits();
    let mut aggregate = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: vec![GradientUpdate {
            layer_id: "a".into(),
            gradients: vec![0.0; 4],
            eligibility_trace: Some(vec![0.0; 4]),
        }],
    };
    let tight = ProtocolLimits {
        max_aggregate_records: 9,
        ..limits
    };
    aggregate.gradients[0].gradients.push(0.0);
    let err = aggregate.validate_with(tight).unwrap_err();
    assert_limit_exceeded(&err, "aggregate");
}
