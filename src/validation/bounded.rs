// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use super::checks::check_finite;
use super::error::ValidationError;

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
#[path = "bounded_tests.rs"]
mod tests;
