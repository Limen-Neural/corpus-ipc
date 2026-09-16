// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed wire-payload validation and resource limits.
//!
//! [`Validate::validate`] and [`Validate::validate_with`] are the single
//! policy used for both direct Rust construction and `TryFrom` deserialization
//! of public wire payloads. Callers that already hold a value (struct
//! literal, tests, a decoder that does not run `Deserialize`) invoke the same
//! checks that JSON/binary ingress uses.
//!
//! # Allocation behavior
//!
//! Deserialize implementations reject sequences and maps once
//! [`ProtocolLimits::DEFAULT`] would be exceeded, and they never honor a
//! serde `size_hint` above that cap. That stops `Vec`/`HashMap` capacity from
//! being used as a denial-of-service primitive.
//!
//! The current encodings still allocate:
//! - each successfully decoded element up to the limit (and at most one
//!   extra element while detecting overflow, via [`serde::de::IgnoredAny`])
//! - each string's bytes as the parser produces them (`serde_json` allocates
//!   the full string before field `Deserialize` runs)
//! - the input buffer itself
//!
//! Preventing those allocations requires a streaming decoder, which this crate
//! does not ship.

mod bounded;
mod checks;
mod error;

pub use error::{ValidationError, ValidationKind, ValidationMeasure};

pub(crate) use bounded::{
    bounded_map, bounded_opt_string, bounded_opt_vec, bounded_string, bounded_vec, finite_f32_at,
};
pub(crate) use checks::{
    add_to_total, check_count, check_finite, check_finite_slice, check_opt_string, check_range,
    check_string, check_unique_by, check_unique_strings,
};

/// Resource caps applied to caller-controlled wire collections and strings.
///
/// Deserialization always uses [`ProtocolLimits::DEFAULT`]. Directly
/// constructed values may pass a different policy to
/// [`Validate::validate_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolLimits {
    /// Max length of stimulus `values`, embeddings, masks, per-update gradient
    /// vectors, eligibility traces, and `ConfigValue::FloatArray`.
    pub max_channel_values: usize,
    /// Max length of `SpikeBatch.spikes`.
    pub max_spike_events: usize,
    /// Max length of `TraceBatch.traces`.
    pub max_traces: usize,
    /// Max length of `GradientBatch.gradients` (number of [`crate::GradientUpdate`] rows).
    pub max_gradients: usize,
    /// Max keys in `BatchMetadata.custom` and `ConfigPayload.config`.
    pub max_metadata_entries: usize,
    /// Max UTF-8 bytes of any protocol string (session ids, keys, values, `source`, `layer_id`).
    pub max_string_bytes: usize,
    /// Max nested records aggregated inside one payload (rows plus inner vectors).
    pub max_aggregate_records: usize,
}

impl ProtocolLimits {
    /// Default wire policy.
    ///
    /// These caps are large enough for current fixtures and typical batch sizes,
    /// and small enough that a `max + 1` test stays a cheap allocation.
    pub const DEFAULT: Self = Self {
        max_channel_values: 65_536,
        max_spike_events: 65_536,
        max_traces: 65_536,
        max_gradients: 65_536,
        max_metadata_entries: 256,
        max_string_bytes: 1_024,
        max_aggregate_records: 131_072,
    };
}

impl Default for ProtocolLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One validation path for public wire payloads.
pub trait Validate {
    /// Validate against [`ProtocolLimits::DEFAULT`].
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_with(ProtocolLimits::DEFAULT)
    }

    /// Validate against an explicit policy.
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_as_str_is_stable() {
        assert_eq!(ValidationKind::NonFinite.as_str(), "non_finite");
        assert_eq!(ValidationKind::OutOfRange.as_str(), "out_of_range");
        assert_eq!(ValidationKind::LengthMismatch.as_str(), "length_mismatch");
        assert_eq!(ValidationKind::LimitExceeded.as_str(), "limit_exceeded");
        assert_eq!(ValidationKind::NestedMetadata.as_str(), "nested_metadata");
        assert_eq!(
            ValidationKind::DuplicateIdentifier.as_str(),
            "duplicate_identifier"
        );
    }

    #[test]
    fn display_includes_path_kind_actual_and_limit() {
        let err = ValidationError::limit_exceeded("spikes", 9, 8);
        let text = err.to_string();
        assert!(text.contains("limit_exceeded"));
        assert!(text.contains("spikes"));
        assert!(text.contains("actual=9"));
        assert!(text.contains("limit=8"));
    }

    #[test]
    fn default_limits_match_associated_constant() {
        assert_eq!(ProtocolLimits::default(), ProtocolLimits::DEFAULT);
    }

    #[test]
    fn nan_measures_compare_by_bits() {
        let a = ValidationMeasure::Float(f32::NAN);
        let b = ValidationMeasure::Float(f32::from_bits(f32::NAN.to_bits()));
        assert_eq!(a, b);
    }

    #[test]
    fn measure_count_and_bytes_are_distinct() {
        assert_ne!(ValidationMeasure::Count(1), ValidationMeasure::Bytes(1));
    }

    #[test]
    fn measure_range_display_is_stable() {
        assert_eq!(
            ValidationMeasure::Range { min: 0.5, max: 2.0 }.to_string(),
            "[0.5, 2]"
        );
    }

    #[test]
    fn display_omits_missing_actual_or_limit() {
        let only_actual = ValidationError::non_finite("x", f32::INFINITY);
        assert!(only_actual.to_string().contains("actual="));
        let neither = ValidationError::nested_metadata("custom");
        assert_eq!(neither.to_string(), "nested_metadata at `custom`");
        let only_limit = ValidationError {
            path: "q".into(),
            kind: ValidationKind::OutOfRange,
            actual: None,
            limit: Some(ValidationMeasure::Range { min: 0.0, max: 1.0 }),
        };
        assert!(only_limit.to_string().contains("limit="));
        assert!(!only_limit.to_string().contains("actual="));
    }
}
