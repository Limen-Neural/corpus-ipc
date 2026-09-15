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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

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
    /// A collection length or string byte length exceeded [`ProtocolLimits`].
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

pub(crate) fn usize_as_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

pub(crate) fn check_string(
    path: &str,
    value: &str,
    limits: ProtocolLimits,
) -> Result<(), ValidationError> {
    if value.len() > limits.max_string_bytes {
        return Err(ValidationError::byte_limit(
            path,
            value.len(),
            limits.max_string_bytes,
        ));
    }
    Ok(())
}

pub(crate) fn check_opt_string(
    path: &str,
    value: Option<&str>,
    limits: ProtocolLimits,
) -> Result<(), ValidationError> {
    match value {
        Some(s) => check_string(path, s, limits),
        None => Ok(()),
    }
}

pub(crate) fn check_count(path: &str, actual: usize, limit: usize) -> Result<(), ValidationError> {
    if actual > limit {
        Err(ValidationError::limit_exceeded(path, actual, limit))
    } else {
        Ok(())
    }
}

pub(crate) fn check_finite(path: &str, value: f32) -> Result<(), ValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ValidationError::non_finite(path, value))
    }
}

pub(crate) fn check_finite_slice(path: &str, values: &[f32]) -> Result<(), ValidationError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(ValidationError::non_finite(
                format!("{path}[{index}]"),
                *value,
            ));
        }
    }
    Ok(())
}

pub(crate) fn check_range(
    path: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), ValidationError> {
    check_finite(path, value)?;
    if value < min || value > max {
        Err(ValidationError::out_of_range(path, value, min, max))
    } else {
        Ok(())
    }
}

pub(crate) fn add_to_total(
    total: &mut usize,
    n: usize,
    limits: ProtocolLimits,
    path: &str,
) -> Result<(), ValidationError> {
    let next = total.checked_add(n).ok_or_else(|| {
        ValidationError::limit_exceeded(path, usize::MAX, limits.max_aggregate_records)
    })?;
    if next > limits.max_aggregate_records {
        return Err(ValidationError::limit_exceeded(
            path,
            next,
            limits.max_aggregate_records,
        ));
    }
    *total = next;
    Ok(())
}

pub(crate) fn check_unique_by<T, K, F>(
    path: &str,
    items: &[T],
    mut key_of: F,
) -> Result<(), ValidationError>
where
    K: Eq + Hash + fmt::Display,
    F: FnMut(&T) -> K,
{
    let mut seen = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let key = key_of(item);
        if !seen.insert(key) {
            let key = key_of(item);
            return Err(ValidationError::duplicate_identifier(
                format!("{path}[{index}]"),
                key,
            ));
        }
    }
    Ok(())
}

fn reject_seq_overflow<'de, A>(seq: &mut A, max: usize, path: &'static str) -> Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    match seq.next_element::<IgnoredAny>() {
        Ok(Some(_)) => {
            let actual = max.saturating_add(1);
            Err(serde::de::Error::custom(ValidationError::limit_exceeded(
                path, actual, max,
            )))
        }
        Ok(None) => Ok(()),
        Err(err) => Err(err),
    }
}

fn reject_map_overflow<'de, A>(
    access: &mut A,
    max_entries: usize,
    path: &'static str,
) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    match access.next_entry::<IgnoredAny, IgnoredAny>() {
        Ok(Some(_)) => {
            let actual = max_entries.saturating_add(1);
            Err(serde::de::Error::custom(ValidationError::limit_exceeded(
                path,
                actual,
                max_entries,
            )))
        }
        Ok(None) => Ok(()),
        Err(err) => Err(err),
    }
}

fn accept_map_key<V, E>(
    map: &HashMap<String, V>,
    key: &str,
    max_key_bytes: usize,
    path: &'static str,
) -> Result<(), E>
where
    E: serde::de::Error,
{
    if key.is_empty() {
        return Err(E::custom(ValidationError::nested_metadata(format!(
            "{path}.<empty>"
        ))));
    }
    if key.len() > max_key_bytes {
        return Err(E::custom(ValidationError::byte_limit(
            format!("{path}.<key>"),
            key.len(),
            max_key_bytes,
        )));
    }
    if map.contains_key(key) {
        return Err(E::custom(ValidationError::duplicate_identifier(
            format!("{path}.{key}"),
            key,
        )));
    }
    Ok(())
}

pub(crate) fn check_unique_strings<T>(
    path: &str,
    items: &[T],
    key_of: impl Fn(&T) -> &str,
) -> Result<(), ValidationError> {
    let mut seen = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let key = key_of(item);
        if !seen.insert(key) {
            return Err(ValidationError::duplicate_identifier(
                format!("{path}[{index}]"),
                key,
            ));
        }
    }
    Ok(())
}

/// Deserialize a sequence, rejecting a length above `max` without trusting `size_hint`.
pub(crate) fn bounded_vec<'de, T, D>(
    deserializer: D,
    max: usize,
    path: &'static str,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct BoundedVecVisitor<T> {
        max: usize,
        path: &'static str,
        _ty: PhantomData<T>,
    }

    impl<'de, T: Deserialize<'de>> Visitor<'de> for BoundedVecVisitor<T> {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a sequence of at most {} items", self.max)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<T>, A::Error> {
            let cap = seq.size_hint().unwrap_or(0).min(self.max);
            let mut out = Vec::with_capacity(cap);
            loop {
                if out.len() >= self.max {
                    reject_seq_overflow(&mut seq, self.max, self.path)?;
                    break;
                }
                match seq.next_element()? {
                    Some(item) => out.push(item),
                    None => break,
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor {
        max,
        path,
        _ty: PhantomData,
    })
}

pub(crate) fn bounded_opt_vec<'de, T, D>(
    deserializer: D,
    max: usize,
    path: &'static str,
) -> Result<Option<Vec<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct OptVisitor<T> {
        max: usize,
        path: &'static str,
        _ty: PhantomData<T>,
    }

    impl<'de, T: Deserialize<'de>> Visitor<'de> for OptVisitor<T> {
        type Value = Option<Vec<T>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "null or a sequence of at most {} items", self.max)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D2: Deserializer<'de>>(
            self,
            deserializer: D2,
        ) -> Result<Self::Value, D2::Error> {
            bounded_vec(deserializer, self.max, self.path).map(Some)
        }
    }

    deserializer.deserialize_option(OptVisitor {
        max,
        path,
        _ty: PhantomData,
    })
}

pub(crate) fn bounded_string<'de, D>(
    deserializer: D,
    max: usize,
    path: &'static str,
) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringVisitor {
        max: usize,
        path: &'static str,
    }

    impl Visitor<'_> for StringVisitor {
        type Value = String;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a string of at most {} bytes", self.max)
        }

        fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<String, E> {
            if value.len() > self.max {
                return Err(E::custom(ValidationError::byte_limit(
                    self.path,
                    value.len(),
                    self.max,
                )));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E: serde::de::Error>(self, value: String) -> Result<String, E> {
            if value.len() > self.max {
                return Err(E::custom(ValidationError::byte_limit(
                    self.path,
                    value.len(),
                    self.max,
                )));
            }
            Ok(value)
        }

        fn visit_bytes<E: serde::de::Error>(self, value: &[u8]) -> Result<String, E> {
            let s = std::str::from_utf8(value).map_err(E::custom)?;
            self.visit_str(s)
        }
    }

    deserializer.deserialize_string(StringVisitor { max, path })
}

pub(crate) fn bounded_opt_string<'de, D>(
    deserializer: D,
    max: usize,
    path: &'static str,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor {
        max: usize,
        path: &'static str,
    }

    impl<'de> Visitor<'de> for OptVisitor {
        type Value = Option<String>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "null or a string of at most {} bytes", self.max)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
            bounded_string_from_visitor(value, self.max, self.path)
                .map_err(E::custom)
                .map(Some)
        }

        fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
            if value.len() > self.max {
                return Err(E::custom(ValidationError::byte_limit(
                    self.path,
                    value.len(),
                    self.max,
                )));
            }
            Ok(Some(value))
        }

        fn visit_some<D2: Deserializer<'de>>(
            self,
            deserializer: D2,
        ) -> Result<Self::Value, D2::Error> {
            bounded_string(deserializer, self.max, self.path).map(Some)
        }
    }

    deserializer.deserialize_option(OptVisitor { max, path })
}

fn bounded_string_from_visitor(
    value: &str,
    max: usize,
    path: &'static str,
) -> Result<String, ValidationError> {
    if value.len() > max {
        Err(ValidationError::byte_limit(path, value.len(), max))
    } else {
        Ok(value.to_owned())
    }
}

/// Deserialize a string-keyed map, bounding entries/keys and rejecting duplicates.
pub(crate) fn bounded_map<'de, V, D>(
    deserializer: D,
    max_entries: usize,
    max_key_bytes: usize,
    path: &'static str,
) -> Result<HashMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct MapVisitor<V> {
        max_entries: usize,
        max_key_bytes: usize,
        path: &'static str,
        _ty: PhantomData<V>,
    }

    impl<'de, V: Deserialize<'de>> Visitor<'de> for MapVisitor<V> {
        type Value = HashMap<String, V>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a map of at most {} entries", self.max_entries)
        }

        fn visit_map<A: MapAccess<'de>>(
            self,
            mut access: A,
        ) -> Result<HashMap<String, V>, A::Error> {
            let cap = access.size_hint().unwrap_or(0).min(self.max_entries);
            let mut map = HashMap::with_capacity(cap);
            loop {
                if map.len() >= self.max_entries {
                    reject_map_overflow(&mut access, self.max_entries, self.path)?;
                    break;
                }
                match access.next_entry::<String, V>()? {
                    Some((key, value)) => {
                        accept_map_key(&map, &key, self.max_key_bytes, self.path)?;
                        map.insert(key, value);
                    }
                    None => break,
                }
            }
            Ok(map)
        }
    }

    deserializer.deserialize_map(MapVisitor {
        max_entries,
        max_key_bytes,
        path,
        _ty: PhantomData,
    })
}

pub(crate) fn finite_f32_at<'de, D>(deserializer: D, path: &'static str) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = f32::deserialize(deserializer)?;
    check_finite(path, value).map_err(serde::de::Error::custom)?;
    Ok(value)
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
    fn add_to_total_rejects_usize_overflow_without_panic() {
        let mut total = usize::MAX;
        let err = add_to_total(&mut total, 1, ProtocolLimits::DEFAULT, "aggregate")
            .expect_err("overflow must be a typed error");
        assert_eq!(err.kind, ValidationKind::LimitExceeded);
        assert_eq!(err.path, "aggregate");
    }

    #[test]
    fn add_to_total_rejects_max_plus_one() {
        let limits = ProtocolLimits {
            max_aggregate_records: 4,
            ..ProtocolLimits::DEFAULT
        };
        let mut total = 4;
        let err = add_to_total(&mut total, 1, limits, "aggregate").unwrap_err();
        assert_eq!(err.kind, ValidationKind::LimitExceeded);
        assert_eq!(err.actual, Some(ValidationMeasure::Count(5)));
        assert_eq!(err.limit, Some(ValidationMeasure::Count(4)));
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
    fn measure_display_and_inequality() {
        assert_ne!(ValidationMeasure::Count(1), ValidationMeasure::Bytes(1));
        assert_eq!(
            ValidationMeasure::Range { min: 0.5, max: 2.0 }.to_string(),
            "[0.5, 2]"
        );
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

    #[test]
    fn bounded_vec_and_map_cover_overflow_and_exact_max() {
        let exact: Vec<u32> = bounded_vec(serde_json::json!([1, 2]), 2, "items").unwrap();
        assert_eq!(exact, vec![1, 2]);
        let err = bounded_vec::<u32, _>(serde_json::json!([1, 2, 3]), 2, "items").unwrap_err();
        assert!(err.to_string().contains("limit_exceeded"));
        let type_err = bounded_vec::<u32, _>(serde_json::json!("nope"), 2, "items").unwrap_err();
        assert!(type_err.to_string().contains("at most 2"));

        let null_mask: Option<Vec<bool>> =
            bounded_opt_vec(serde_json::Value::Null, 4, "valid_mask").unwrap();
        assert!(null_mask.is_none());
        let some_mask: Option<Vec<bool>> =
            bounded_opt_vec(serde_json::json!([true, false]), 4, "valid_mask").unwrap();
        assert_eq!(some_mask, Some(vec![true, false]));

        let ok = bounded_string(serde_json::Value::String("ab".into()), 2, "s").unwrap();
        assert_eq!(ok, "ab");
        let long = bounded_string(serde_json::Value::String("abc".into()), 2, "s").unwrap_err();
        assert!(long.to_string().contains("limit_exceeded"));
        let over_str = bounded_string(
            serde::de::value::BorrowedStrDeserializer::<serde::de::value::Error>::new("abcdef"),
            2,
            "s",
        )
        .unwrap_err();
        assert!(over_str.to_string().contains("limit_exceeded"));
        let bytes = bounded_string(
            serde::de::value::BorrowedBytesDeserializer::<serde::de::value::Error>::new(b"abc"),
            2,
            "s",
        )
        .unwrap_err();
        assert!(bytes.to_string().contains("limit_exceeded"));

        let map: std::collections::HashMap<String, String> =
            bounded_map(serde_json::json!({"a": "1", "b": "2"}), 2, 8, "custom").unwrap();
        assert_eq!(map.len(), 2);
        let overflow = bounded_map::<String, _>(
            serde_json::json!({"a": "1", "b": "2", "c": "3"}),
            2,
            8,
            "custom",
        )
        .unwrap_err();
        assert!(overflow.to_string().contains("limit_exceeded"));
        let empty_key =
            bounded_map::<String, _>(serde_json::json!({"": "v"}), 8, 8, "custom").unwrap_err();
        assert!(empty_key.to_string().contains("nested_metadata"));
        let long_key = bounded_map::<String, _>(serde_json::json!({"abcdef": "v"}), 8, 3, "custom")
            .unwrap_err();
        assert!(long_key.to_string().contains("limit_exceeded"));
        let map_ty = bounded_map::<String, _>(serde_json::json!([1]), 8, 8, "custom").unwrap_err();
        assert!(map_ty.to_string().contains("at most 8"));

        let inf = finite_f32_at(serde_json::json!(1e39), "value").unwrap_err();
        assert!(inf.to_string().contains("non_finite"));
        assert!(finite_f32_at(serde_json::json!(1.5), "value").is_ok());
    }

    #[test]
    fn duplicate_string_keys_and_unique_helper() {
        struct Row {
            id: String,
        }
        let items = [Row { id: "a".into() }, Row { id: "a".into() }];
        let err = check_unique_strings("gradients", &items, |row| row.id.as_str()).unwrap_err();
        assert_eq!(err.kind, ValidationKind::DuplicateIdentifier);
        check_unique_strings("gradients", &items[..1], |row| row.id.as_str()).unwrap();
    }
}
