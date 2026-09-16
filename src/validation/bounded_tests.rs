// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use std::collections::HashMap;

#[test]
fn bounded_vec_accepts_exact_max() {
    let exact: Vec<u32> = bounded_vec(serde_json::json!([1, 2]), 2, "items").unwrap();
    assert_eq!(exact, vec![1, 2]);
}

#[test]
fn bounded_vec_rejects_overflow() {
    let err = bounded_vec::<u32, _>(serde_json::json!([1, 2, 3]), 2, "items").unwrap_err();
    assert!(err.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_vec_rejects_wrong_type() {
    let err = bounded_vec::<u32, _>(serde_json::json!("nope"), 2, "items").unwrap_err();
    assert!(err.to_string().contains("at most 2"));
}

#[test]
fn bounded_opt_vec_accepts_null_and_values() {
    let null_mask: Option<Vec<bool>> =
        bounded_opt_vec(serde_json::Value::Null, 4, "valid_mask").unwrap();
    assert!(null_mask.is_none());
    let some_mask: Option<Vec<bool>> =
        bounded_opt_vec(serde_json::json!([true, false]), 4, "valid_mask").unwrap();
    assert_eq!(some_mask, Some(vec![true, false]));
}

#[test]
fn bounded_string_accepts_exact_max_and_rejects_overflow() {
    let ok = bounded_string(serde_json::Value::String("ab".into()), 2, "s").unwrap();
    assert_eq!(ok, "ab");
    let long = bounded_string(serde_json::Value::String("abc".into()), 2, "s").unwrap_err();
    assert!(long.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_string_str_visitor_rejects_overflow() {
    let over_str = bounded_string(
        serde::de::value::BorrowedStrDeserializer::<serde::de::value::Error>::new("abcdef"),
        2,
        "s",
    )
    .unwrap_err();
    assert!(over_str.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_string_bytes_visitor_rejects_overflow() {
    let bytes = bounded_string(
        serde::de::value::BorrowedBytesDeserializer::<serde::de::value::Error>::new(b"abc"),
        2,
        "s",
    )
    .unwrap_err();
    assert!(bytes.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_opt_string_null_and_overflow() {
    let none: Option<String> =
        bounded_opt_string(serde_json::Value::Null, 8, "session_id").unwrap();
    assert!(none.is_none());
    let over = bounded_opt_string(serde_json::Value::String("toolong".into()), 2, "session_id")
        .unwrap_err();
    assert!(over.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_map_accepts_exact_max() {
    let map: HashMap<String, String> =
        bounded_map(serde_json::json!({"a": "1", "b": "2"}), 2, 8, "custom").unwrap();
    assert_eq!(map.len(), 2);
}

#[test]
fn bounded_map_rejects_overflow() {
    let overflow = bounded_map::<String, _>(
        serde_json::json!({"a": "1", "b": "2", "c": "3"}),
        2,
        8,
        "custom",
    )
    .unwrap_err();
    assert!(overflow.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_map_rejects_empty_key() {
    let empty_key =
        bounded_map::<String, _>(serde_json::json!({"": "v"}), 8, 8, "custom").unwrap_err();
    assert!(empty_key.to_string().contains("nested_metadata"));
}

#[test]
fn bounded_map_rejects_oversize_key() {
    let long_key =
        bounded_map::<String, _>(serde_json::json!({"abcdef": "v"}), 8, 3, "custom").unwrap_err();
    assert!(long_key.to_string().contains("limit_exceeded"));
}

#[test]
fn bounded_map_rejects_wrong_type() {
    let map_ty = bounded_map::<String, _>(serde_json::json!([1]), 8, 8, "custom").unwrap_err();
    assert!(map_ty.to_string().contains("at most 8"));
}

#[test]
fn finite_f32_rejects_infinity_and_accepts_finite() {
    let inf = finite_f32_at(serde_json::json!(1e39), "value").unwrap_err();
    assert!(inf.to_string().contains("non_finite"));
    assert!(finite_f32_at(serde_json::json!(1.5), "value").is_ok());
}
