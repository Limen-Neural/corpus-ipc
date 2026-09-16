// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// Stable classification of a protocol validation failure.
///
/// These identifiers are part of the crate's error contract: serde wrappers
/// and [`fmt::Display`] use [`ValidationKind::as_str`], so assertions should
/// match the kind rather than a full prose sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationKind {
    /// An `f32` was NaN or ±infinity.
    NonFinite,
    /// A numeric value was outside its documented inclusive range.
    OutOfRange,
    /// Two related sequences (for example `values` and `valid_mask`) differed in length.
    LengthMismatch,
    /// A collection length or string byte length exceeded [`ProtocolLimits`](crate::ProtocolLimits).
    LimitExceeded,
    /// Nested metadata violated policy (empty key, or a nested object rejected
    /// by the typed map).
    NestedMetadata,
    /// A repeated identifier in a list that must be unique (`channel_id`, `layer_id`).
    DuplicateIdentifier,
}

impl ValidationKind {
    /// Stable machine-readable kind name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonFinite => "non_finite",
            Self::OutOfRange => "out_of_range",
            Self::LengthMismatch => "length_mismatch",
            Self::LimitExceeded => "limit_exceeded",
            Self::NestedMetadata => "nested_metadata",
            Self::DuplicateIdentifier => "duplicate_identifier",
        }
    }
}

impl fmt::Display for ValidationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Observed count, byte length, float, or documented range attached to a
/// [`ValidationError`].
#[derive(Debug, Clone, Copy)]
pub enum ValidationMeasure {
    /// Collection length or identifier code.
    Count(u64),
    /// UTF-8 byte length of a string.
    Bytes(u64),
    /// Observed `f32` (compared by bits; may be NaN or ±Inf).
    Float(f32),
    /// Inclusive documented range that a float must occupy.
    Range { min: f32, max: f32 },
}

impl PartialEq for ValidationMeasure {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Count(a), Self::Count(b)) | (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (
                Self::Range {
                    min: a_min,
                    max: a_max,
                },
                Self::Range {
                    min: b_min,
                    max: b_max,
                },
            ) => a_min.to_bits() == b_min.to_bits() && a_max.to_bits() == b_max.to_bits(),
            _ => false,
        }
    }
}

impl Eq for ValidationMeasure {}

impl fmt::Display for ValidationMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count(n) | Self::Bytes(n) => write!(f, "{n}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Range { min, max } => write!(f, "[{min}, {max}]"),
        }
    }
}

/// Typed validation failure for a wire payload.
///
/// `path` is a JSON-like field path (`tempo`, `spikes`, `traces[1].channel_id`).
/// `actual` / `limit` identify the observed value or size and the applicable
/// bound when those are meaningful for [`kind`](Self::kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    /// Field or indexed path that failed.
    pub path: String,
    /// Stable error kind.
    pub kind: ValidationKind,
    /// Observed value, count, or byte length when applicable.
    pub actual: Option<ValidationMeasure>,
    /// Applicable limit or documented range.
    pub limit: Option<ValidationMeasure>,
}

impl ValidationError {
    /// Non-finite float at `path`.
    #[must_use]
    pub fn non_finite(path: impl Into<String>, actual: f32) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::NonFinite,
            actual: Some(ValidationMeasure::Float(actual)),
            limit: None,
        }
    }

    /// Float outside `[min, max]` at `path`.
    #[must_use]
    pub fn out_of_range(path: impl Into<String>, actual: f32, min: f32, max: f32) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::OutOfRange,
            actual: Some(ValidationMeasure::Float(actual)),
            limit: Some(ValidationMeasure::Range { min, max }),
        }
    }

    /// Related sequences had different lengths.
    #[must_use]
    pub fn length_mismatch(path: impl Into<String>, actual: usize, expected: usize) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::LengthMismatch,
            actual: Some(ValidationMeasure::Count(usize_as_u64(actual))),
            limit: Some(ValidationMeasure::Count(usize_as_u64(expected))),
        }
    }

    /// Collection length exceeded `limit`.
    #[must_use]
    pub fn limit_exceeded(path: impl Into<String>, actual: usize, limit: usize) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::LimitExceeded,
            actual: Some(ValidationMeasure::Count(usize_as_u64(actual))),
            limit: Some(ValidationMeasure::Count(usize_as_u64(limit))),
        }
    }

    /// String byte length exceeded `limit`.
    #[must_use]
    pub fn byte_limit(path: impl Into<String>, actual: usize, limit: usize) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::LimitExceeded,
            actual: Some(ValidationMeasure::Bytes(usize_as_u64(actual))),
            limit: Some(ValidationMeasure::Bytes(usize_as_u64(limit))),
        }
    }

    /// Nested metadata policy violation (for example an empty key).
    #[must_use]
    pub fn nested_metadata(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::NestedMetadata,
            actual: None,
            limit: None,
        }
    }

    /// Duplicate identifier at `path`.
    #[must_use]
    pub fn duplicate_identifier(path: impl Into<String>, identifier: impl fmt::Display) -> Self {
        Self {
            path: path.into(),
            kind: ValidationKind::DuplicateIdentifier,
            actual: Some(ValidationMeasure::Count(0)),
            limit: None,
        }
        .with_duplicate_display(identifier)
    }

    fn with_duplicate_display(mut self, identifier: impl fmt::Display) -> Self {
        // Keep `actual` as a count when the identifier is numeric; otherwise the
        // path carries the identity and `actual` stays unused.
        let label = identifier.to_string();
        if let Ok(n) = label.parse::<u64>() {
            self.actual = Some(ValidationMeasure::Count(n));
        } else {
            self.actual = None;
            if !self.path.contains(&label) {
                self.path = format!("{}:{label}", self.path);
            }
        }
        self
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at `{}`", self.kind, self.path)?;
        match (&self.actual, &self.limit) {
            (Some(actual), Some(limit)) => write!(f, ", actual={actual}, limit={limit}"),
            (Some(actual), None) => write!(f, ", actual={actual}"),
            (None, Some(limit)) => write!(f, ", limit={limit}"),
            (None, None) => Ok(()),
        }
    }
}

impl std::error::Error for ValidationError {}
pub(crate) fn usize_as_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}
