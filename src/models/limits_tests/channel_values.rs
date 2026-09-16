// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn stimulus_values_accept_exact_max() {
    let limits = limits();
    let stimulus = StimulusBatch {
        values: vec![0.0; limits.max_channel_values],
        ..Default::default()
    };
    stimulus.validate_with(limits).expect("channel max");
}

#[test]
fn stimulus_values_reject_max_plus_one() {
    let limits = limits();
    let mut stimulus = StimulusBatch {
        values: vec![0.0; limits.max_channel_values],
        ..Default::default()
    };
    stimulus.values.push(0.0);
    let err = stimulus.validate_with(limits).unwrap_err();
    assert_limit_exceeded(&err, "values");
    assert_eq!(
        err.actual,
        Some(ValidationMeasure::Count(
            u64::try_from(limits.max_channel_values + 1).unwrap()
        ))
    );
    assert_eq!(
        err.limit,
        Some(ValidationMeasure::Count(
            u64::try_from(limits.max_channel_values).unwrap()
        ))
    );
}

#[test]
fn mask_value_length_mismatch_is_typed() {
    let batch = StimulusBatch {
        values: vec![0.0, 1.0],
        valid_mask: Some(vec![true]),
        ..Default::default()
    };
    let err = batch.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::LengthMismatch);
    assert_eq!(err.actual, Some(ValidationMeasure::Count(1)));
    assert_eq!(err.limit, Some(ValidationMeasure::Count(2)));
}
