// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wire-v1 golden-vector corpus tests (LIM-1446 / RM-1330).
//!
//! `test-vectors/wire-v1/` is the published golden corpus for the canonical
//! wire-v1 encoding (`encode_canonical_ipc_message`, `docs/wire-encoding.md`).
//!
//! - **Positive** vectors are *generated* by the canonical encoder and must
//!   stay byte-stable: the file bytes must equal a fresh encode of the same
//!   message, and decode/re-encode must round-trip to identical bytes.
//! - **Legacy**, **alias**, and **normalization** vectors are hand-maintained
//!   decode targets: they record the expected decoded `IpcMessage` (including
//!   the documented `ConfigValue::Integer` -> `ConfigValue::Float`
//!   normalization and `neuron_id` -> `channel_id` alias). They must never be
//!   regenerated through the canonical encoder.
//! - **Negative** vectors record the expected failure class (too old, too new,
//!   payload error, JSON error). Unsupported-version vectors prove the version
//!   check runs before payload interpretation.
//!
//! Fixture drift on a positive vector means either the fixture or the encoder
//! changed. Regenerate intentionally with:
//!
//! ```bash
//! REGENERATE_WIRE_V1=1 cargo test --locked --test wire_v1_regenerate
//! ```

mod shared;

use std::collections::{HashMap, HashSet};

use corpus_ipc::{
    CompatibilityError, ConfigValue, EnvelopeError, IpcMessage, WireCompatibility,
    decode_ipc_message_json, encode_canonical_ipc_message,
};
use shared::{
    EXPECTED_VARIANTS, ManifestEntry, decode_vectors, fixture_bytes, load_manifest,
    positive_vectors, sha256_hex,
};

#[test]
fn manifest_is_well_formed_and_digests_match() {
    let manifest = load_manifest();
    assert_eq!(manifest.schema_version, 1);
    assert!(
        manifest.encoding_profile.contains("wire v1"),
        "profile must name the wire-v1 encoding: {}",
        manifest.encoding_profile
    );
    assert!(
        manifest.generator.contains("wire_v1_regenerate"),
        "manifest must record the regeneration command"
    );
    let mut seen = HashSet::new();
    for entry in &manifest.fixtures {
        assert!(
            seen.insert(entry.name.as_str()),
            "duplicate fixture name {}",
            entry.name
        );
        let bytes = fixture_bytes(entry);
        assert_eq!(
            sha256_hex(&bytes),
            entry.sha256,
            "sha256 drift on {}",
            entry.name
        );
    }
}

#[test]
fn positive_corpus_covers_every_variant() {
    let manifest = load_manifest();
    let covered: HashSet<&str> = manifest
        .fixtures
        .iter()
        .filter(|f| f.kind == "positive")
        .filter_map(|f| f.variant.as_deref())
        .collect();
    for variant in EXPECTED_VARIANTS {
        assert!(
            covered.contains(variant),
            "missing positive vector for IpcMessage::{variant}"
        );
    }
    // Every constructor entry is distinct and names a real variant.
    let mut files = HashSet::new();
    for (name, variant, _) in positive_vectors() {
        assert!(files.insert(name), "duplicate positive name {name}");
        assert!(
            EXPECTED_VARIANTS.contains(&variant),
            "{name} names unknown variant {variant}"
        );
    }
}

#[test]
fn positive_vectors_are_byte_stable_and_round_trip() {
    let manifest = load_manifest();
    let by_name: HashMap<&str, &ManifestEntry> = manifest
        .fixtures
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    for (name, variant, message) in positive_vectors() {
        check_positive(&by_name, name, variant, &message);
    }
}

fn check_positive(
    by_name: &HashMap<&str, &ManifestEntry>,
    name: &str,
    variant: &str,
    message: &IpcMessage,
) {
    let entry = by_name
        .get(name)
        .unwrap_or_else(|| panic!("manifest lacks positive {name}"));
    assert_eq!(entry.kind, "positive", "{name} kind");
    assert_eq!(entry.variant.as_deref(), Some(variant), "{name} variant");
    assert_eq!(entry.expect, "decode_ok", "{name} expectation");

    let on_disk = fixture_bytes(entry);
    // Byte-stable regeneration: file bytes equal a fresh canonical encode.
    let regenerated =
        encode_canonical_ipc_message(message).unwrap_or_else(|e| panic!("encode {name}: {e}"));
    assert_eq!(
        on_disk,
        regenerated,
        "{name} drifted; regenerate with {}",
        shared::REGENERATE_COMMAND
    );
    // Canonical decode/re-encode equivalence.
    let decoded =
        decode_ipc_message_json(&on_disk).unwrap_or_else(|e| panic!("decode {name}: {e}"));
    assert_eq!(
        decoded, *message,
        "{name} decoded value mismatch (note: ConfigValue::Integer normalizes \
         to Float on decode; keep positives inside the normalized domain)"
    );
    let reencoded = encode_canonical_ipc_message(&decoded).unwrap();
    assert_eq!(reencoded, on_disk, "{name} re-encode not identical");
}

#[test]
fn hand_maintained_vectors_decode_to_expected_messages() {
    let manifest = load_manifest();
    let by_name: HashMap<&str, &ManifestEntry> = manifest
        .fixtures
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    for (name, expected) in decode_vectors() {
        let entry = by_name
            .get(name)
            .unwrap_or_else(|| panic!("manifest lacks {name}"));
        assert!(
            matches!(entry.kind.as_str(), "legacy" | "alias" | "normalization"),
            "{name} kind {}",
            entry.kind
        );
        assert_eq!(entry.expect, "decode_ok", "{name} expectation");
        let decoded = decode_ipc_message_json(&fixture_bytes(entry))
            .unwrap_or_else(|e| panic!("decode {name}: {e}"));
        assert_eq!(decoded, expected, "{name} decoded value mismatch");
    }
}

#[test]
fn normalization_vector_documents_integer_to_float_loss() {
    // Explicitly record the non-lossless decode: the on-wire `big` integer is
    // 16_777_217 but decodes as Float(16_777_216.0) (2^24), the closest f32.
    let manifest = load_manifest();
    let entry = manifest
        .fixtures
        .iter()
        .find(|f| f.name == "normalization_integer_config")
        .expect("manifest entry");
    let bytes = fixture_bytes(entry);
    assert!(
        String::from_utf8_lossy(&bytes).contains("16777217"),
        "fixture must keep the literal 2^24+1 input"
    );
    let IpcMessage::ConfigUpdate(payload) = decode_ipc_message_json(&bytes).unwrap() else {
        panic!("expected ConfigUpdate");
    };
    assert_eq!(payload.config.get("n"), Some(&ConfigValue::Float(42.0)));
    match payload.config.get("big") {
        Some(ConfigValue::Float(v)) => {
            assert_eq!(*v, 16_777_216.0);
            assert_ne!(*v as u64, 16_777_217, "precision loss is the contract");
        }
        other => panic!("expected Float after normalization, got {other:?}"),
    }
}

#[test]
fn negative_vectors_fail_with_expected_error_class() {
    let manifest = load_manifest();
    let mut negatives = 0;
    for entry in manifest.fixtures.iter().filter(|f| f.kind == "negative") {
        negatives += 1;
        check_negative(entry);
    }
    assert!(negatives >= 5, "corpus needs negative coverage");
}

fn check_negative(entry: &ManifestEntry) {
    let err = decode_ipc_message_json(&fixture_bytes(entry))
        .expect_err(&format!("{} must fail", entry.name));
    let name = &entry.name;
    match entry.expect.as_str() {
        "too_old" => assert!(
            matches!(
                err,
                EnvelopeError::Compatibility(CompatibilityError::TooOld { .. })
            ),
            "{name}: expected TooOld, got {err}"
        ),
        "too_new" => assert!(
            matches!(
                err,
                EnvelopeError::Compatibility(CompatibilityError::TooNew { .. })
            ),
            "{name}: expected TooNew, got {err}"
        ),
        "payload_error" => assert!(
            matches!(err, EnvelopeError::Payload(_)),
            "{name}: expected payload error, got {err}"
        ),
        "json_error" => assert!(
            matches!(err, EnvelopeError::Json(_)),
            "{name}: expected JSON error, got {err}"
        ),
        other => panic!("{name}: unknown expectation {other}"),
    }
}

#[test]
fn unsupported_version_is_rejected_before_payload_interpretation() {
    // v2 envelope whose payload is *not valid JSON-typed IpcMessage at all*:
    // if the version check did not run first this would surface a payload
    // error instead of TooNew.
    let manifest = load_manifest();
    let entry = manifest
        .fixtures
        .iter()
        .find(|f| f.name == "v2_too_new_malformed_payload")
        .expect("manifest entry");
    let err = decode_ipc_message_json(&fixture_bytes(entry)).unwrap_err();
    assert!(
        matches!(
            err,
            EnvelopeError::Compatibility(CompatibilityError::TooNew { found, .. })
                if found == WireCompatibility::CURRENT + 1
        ),
        "version must be checked before payload use, got {err}"
    );
}

/// Keep fixture files readable: every positive file must be exactly the
/// canonical compact encoding (no trailing newline, no whitespace).
#[test]
fn positive_files_are_compact_canonical_json() {
    let manifest = load_manifest();
    for entry in manifest.fixtures.iter().filter(|f| f.kind == "positive") {
        let bytes = fixture_bytes(entry);
        assert!(
            !bytes.ends_with(b"\n"),
            "{}: canonical JSON has no trailing newline",
            entry.name
        );
        let text = String::from_utf8(bytes).expect("utf8");
        assert_eq!(
            text,
            text.trim(),
            "{}: no surrounding whitespace",
            entry.name
        );
    }
}
