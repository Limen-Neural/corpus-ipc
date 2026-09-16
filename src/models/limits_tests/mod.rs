// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use serde::de::DeserializeOwned;

pub(super) use crate::validation::{
    ProtocolLimits, Validate, ValidationError, ValidationKind, ValidationMeasure,
};

mod aggregate;
mod channel_values;
mod config;
mod dispatch;
mod embeddings;
mod gradients;
mod metadata;
mod nested;
mod non_finite;
mod roundtrip;
mod serde_reject;
mod spikes;
mod strings;
mod traces;

fn limits() -> ProtocolLimits {
    ProtocolLimits::DEFAULT
}

fn oversize_string(limit: usize) -> String {
    "a".repeat(limit.saturating_add(1))
}

fn max_string(limit: usize) -> String {
    "a".repeat(limit)
}

fn serde_err<T: DeserializeOwned>(json: serde_json::Value) -> String {
    match serde_json::from_value::<T>(json) {
        Ok(_) => panic!("payload must be rejected"),
        Err(err) => err.to_string(),
    }
}

fn assert_serde_kind<T: DeserializeOwned>(json: serde_json::Value, kind: ValidationKind) {
    let msg = serde_err::<T>(json);
    assert!(
        msg.contains(kind.as_str()),
        "expected {} in serde error, got {msg}",
        kind.as_str()
    );
}

fn sample_gradient_update() -> GradientUpdate {
    GradientUpdate {
        layer_id: "layer-a".into(),
        gradients: vec![0.1, 0.2],
        eligibility_trace: None,
    }
}

fn non_finite_values() -> [f32; 3] {
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
}

fn assert_limit_exceeded(err: &ValidationError, path: &str) {
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, path);
}

#[test]
fn prefix_path_uses_prefix_when_error_path_is_empty() {
    let mut empty = ValidationError::nested_metadata("");
    empty.path.clear();
    let prefixed = super::de::prefix_path(empty, "root");
    assert_eq!(prefixed.path, "root");
}
