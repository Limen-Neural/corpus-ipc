// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

use super::ProtocolLimits;
use super::error::ValidationError;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::{ProtocolLimits, ValidationKind, ValidationMeasure};

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
    fn unique_strings_reject_duplicates() {
        struct Row {
            id: String,
        }
        let items = [Row { id: "a".into() }, Row { id: "a".into() }];
        let err = check_unique_strings("gradients", &items, |row| row.id.as_str()).unwrap_err();
        assert_eq!(err.kind, ValidationKind::DuplicateIdentifier);
        check_unique_strings("gradients", &items[..1], |row| row.id.as_str()).unwrap();
    }
}
