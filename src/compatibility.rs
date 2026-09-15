// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fail-closed wire-schema compatibility for [`IpcMessage`].
//!
//! Crate semver (`CARGO_PKG_VERSION`) answers whether a consumer may depend on
//! this library. It does **not** answer whether an on-wire payload may be used.
//! [`WireCompatibility`] is the single public source of truth for that window:
//! incoming envelopes are classified as supported, too old, or too new, and
//! decoded only after the version check succeeds.
//!
//! # Compatibility table (wire version 1)
//!
//! | Incoming | Relation | Result |
//! | --- | --- | --- |
//! | `MIN_SUPPORTED - 1` (`0`) | too old | [`CompatibilityError::TooOld`] |
//! | `MIN_SUPPORTED` (`1`) | minimum supported | [`Compatibility::Supported`] |
//! | `CURRENT` (`1`) | current | [`Compatibility::Supported`] |
//! | `CURRENT + 1` (`2`) | too new | [`CompatibilityError::TooNew`] |
//!
//! Unversioned JSON (the pre-envelope `IpcMessage` encoding) is treated as
//! [`WireCompatibility::LEGACY_UNVERSIONED`] (`1`), which is inside the
//! supported window.
//!
//! # Current encoding rules
//!
//! The current encoding is serde JSON with externally tagged [`IpcMessage`]
//! variants (`{"Spikes":{...}}`, `"Ping"`, …). Unit variants encode as JSON
//! strings; struct variants encode as single-key objects.
//!
//! - **Unknown fields** are ignored on structs and on the envelope object.
//!   Older readers can therefore skip additive optional fields without a
//!   wire-version bump. This is the forward-compatibility rule.
//! - **Unknown variants** never become a valid default. [`IpcMessage`] has no
//!   `Default` impl and no `#[serde(other)]` / `#[serde(other = "...")]`
//!   catch-all; an unrecognized variant name fails to deserialize.
//!
//! # Non-goals
//!
//! This module is not a negotiation daemon, schema registry, or transport
//! handshake. It does not promise to decode arbitrary future versions.

use serde::de::{DeserializeOwned, Error as DeError};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::IpcMessage;

/// Canonical wire-schema compatibility window for this crate.
///
/// These constants are independent of crate semver. Bump them only according
/// to the rules in `CHANGELOG.md` (when a wire-version bump is required).
pub struct WireCompatibility;

impl WireCompatibility {
    /// Schema version produced by this crate's envelope encoders.
    pub const CURRENT: u32 = 1;

    /// Oldest schema version this crate still accepts.
    pub const MIN_SUPPORTED: u32 = 1;

    /// Version assigned to pre-envelope `IpcMessage` JSON (no `wire_version`).
    ///
    /// That encoding is the original serde-tagged enum. It remains supported
    /// for as long as `LEGACY_UNVERSIONED` stays inside
    /// `[MIN_SUPPORTED, CURRENT]`.
    pub const LEGACY_UNVERSIONED: u32 = 1;

    /// Classify `version` against [`MIN_SUPPORTED`](Self::MIN_SUPPORTED) and
    /// [`CURRENT`](Self::CURRENT) without decoding a payload.
    #[must_use]
    pub fn classify(version: u32) -> Compatibility {
        if version < Self::MIN_SUPPORTED {
            Compatibility::TooOld
        } else if version > Self::CURRENT {
            Compatibility::TooNew
        } else {
            Compatibility::Supported
        }
    }

    /// Accept `version` or return a typed, actionable error.
    ///
    /// Call this **before** interpreting payload bytes.
    pub fn accept(version: u32) -> Result<SupportedWireVersion, CompatibilityError> {
        match Self::classify(version) {
            Compatibility::Supported => Ok(SupportedWireVersion(version)),
            Compatibility::TooOld => Err(CompatibilityError::TooOld {
                found: version,
                min: Self::MIN_SUPPORTED,
            }),
            Compatibility::TooNew => Err(CompatibilityError::TooNew {
                found: version,
                current: Self::CURRENT,
            }),
        }
    }
}

/// Result of comparing an incoming version to the supported window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// `MIN_SUPPORTED <= version <= CURRENT`.
    Supported,
    /// `version < MIN_SUPPORTED`.
    TooOld,
    /// `version > CURRENT`.
    TooNew,
}

/// An incoming version that passed [`WireCompatibility::accept`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SupportedWireVersion(u32);

impl SupportedWireVersion {
    /// The accepted wire version.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Classify `version` as supported, too old, or too new.
///
/// This is the public function form of [`WireCompatibility::classify`].
#[must_use]
pub fn classify_wire_version(version: u32) -> Compatibility {
    WireCompatibility::classify(version)
}

/// Accept `version` or return [`CompatibilityError`] before payload use.
pub fn accept_wire_version(version: u32) -> Result<SupportedWireVersion, CompatibilityError> {
    WireCompatibility::accept(version)
}

/// Fail-closed errors for versions outside the supported window.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompatibilityError {
    /// Producer is older than this crate still decodes.
    #[error(
        "wire version {found} is too old (minimum supported is {min}); upgrade the producer to emit version {min} or newer"
    )]
    TooOld {
        /// Version found on the envelope.
        found: u32,
        /// [`WireCompatibility::MIN_SUPPORTED`].
        min: u32,
    },
    /// Producer is newer than this crate understands.
    #[error(
        "wire version {found} is too new (current is {current}); upgrade corpus-ipc or have the producer emit version {current} or older"
    )]
    TooNew {
        /// Version found on the envelope.
        found: u32,
        /// [`WireCompatibility::CURRENT`].
        current: u32,
    },
}

/// Errors from envelope JSON decode, including compatibility failures.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    /// Version is outside the supported window.
    #[error(transparent)]
    Compatibility(#[from] CompatibilityError),
    /// Top-level JSON was not an object.
    #[error("compatibility envelope must be a JSON object")]
    NotAnObject,
    /// Envelope object lacked `wire_version`.
    #[error("missing wire_version on compatibility envelope")]
    MissingVersion,
    /// `wire_version` was present but not a `u32`.
    #[error("wire_version must be a non-negative integer fitting u32, got {0}")]
    InvalidVersion(String),
    /// Envelope object lacked `payload` after the version check passed.
    #[error("missing payload on compatibility envelope")]
    MissingPayload,
    /// Failed to parse the outer JSON document.
    #[error("failed to decode envelope JSON: {0}")]
    Json(serde_json::Error),
    /// Version was supported, but the payload did not decode as `T`.
    #[error("failed to decode envelope payload: {0}")]
    Payload(serde_json::Error),
}

/// Versioned wrapper around a hybrid-flow payload.
///
/// JSON shape:
///
/// ```json
/// {"wire_version":1,"payload":{"Ping":null}}
/// ```
///
/// Transport ownership is unchanged: backends still send and receive bytes.
/// Call [`WireEnvelope::decode_json`] or [`decode_ipc_message_json`] at the
/// decode entry point so too-old / too-new envelopes fail before the payload
/// is used.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WireEnvelope<T> {
    /// Schema version of `payload`.
    pub wire_version: u32,
    /// Typed hybrid-flow message (typically [`IpcMessage`]).
    pub payload: T,
}

impl<T> WireEnvelope<T> {
    /// Wrap `payload` at [`WireCompatibility::CURRENT`].
    #[must_use]
    pub fn new(payload: T) -> Self {
        Self {
            wire_version: WireCompatibility::CURRENT,
            payload,
        }
    }

    /// Unwrap the payload after a successful decode.
    #[must_use]
    pub fn into_payload(self) -> T {
        self.payload
    }
}

impl<T: DeserializeOwned> WireEnvelope<T> {
    /// Decode JSON bytes, checking `wire_version` before converting `payload`.
    pub fn decode_json(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        let value = serde_json::from_slice(bytes).map_err(EnvelopeError::Json)?;
        Self::from_json_value(value)
    }

    /// Decode an already-parsed JSON value, checking version before `T`.
    pub fn from_json_value(value: Value) -> Result<Self, EnvelopeError> {
        let Value::Object(mut obj) = value else {
            return Err(EnvelopeError::NotAnObject);
        };
        let version_val = obj
            .remove("wire_version")
            .ok_or(EnvelopeError::MissingVersion)?;
        let version = parse_wire_version(&version_val)?;
        // Fail closed on the version *before* interpreting payload as `T`.
        WireCompatibility::accept(version)?;
        let payload_val = obj.remove("payload").ok_or(EnvelopeError::MissingPayload)?;
        let payload = serde_json::from_value(payload_val).map_err(EnvelopeError::Payload)?;
        Ok(Self {
            wire_version: version,
            payload,
        })
    }
}

impl<T: Serialize> WireEnvelope<T> {
    /// Encode this envelope as compact JSON bytes.
    pub fn encode_json(&self) -> Result<Vec<u8>, EnvelopeError> {
        serde_json::to_vec(self).map_err(EnvelopeError::Json)
    }
}

impl<'de, T> Deserialize<'de> for WireEnvelope<T>
where
    T: DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_json_value(value).map_err(D::Error::custom)
    }
}

/// Encode `message` as a current-version compatibility envelope.
pub fn encode_ipc_message_json(message: &IpcMessage) -> Result<Vec<u8>, EnvelopeError> {
    WireEnvelope::new(message).encode_json()
}

/// Decode an [`IpcMessage`] from JSON, accepting either an envelope or legacy
/// unversioned tagged JSON.
///
/// Envelope path: `{"wire_version":N,"payload":{...}}` — `N` is classified
/// before the payload is deserialized.
///
/// Legacy path: `{"Spikes":{...}}` or `"Ping"` — treated as
/// [`WireCompatibility::LEGACY_UNVERSIONED`].
///
/// ```
/// use corpus_ipc::{decode_ipc_message_json, encode_ipc_message_json, IpcMessage};
///
/// let bytes = encode_ipc_message_json(&IpcMessage::Ping).unwrap();
/// let message = decode_ipc_message_json(&bytes).unwrap();
/// assert!(matches!(message, IpcMessage::Ping));
/// ```
pub fn decode_ipc_message_json(bytes: &[u8]) -> Result<IpcMessage, EnvelopeError> {
    let value = serde_json::from_slice(bytes).map_err(EnvelopeError::Json)?;
    decode_ipc_message_value(value)
}

/// [`decode_ipc_message_json`] for an already-parsed [`serde_json::Value`].
pub fn decode_ipc_message_value(value: Value) -> Result<IpcMessage, EnvelopeError> {
    match &value {
        // Externally tagged unit variants (`Ping`, `Shutdown`, `TrainingComplete`)
        // serialize as a JSON string, not an object.
        Value::String(_) => {
            WireCompatibility::accept(WireCompatibility::LEGACY_UNVERSIONED)?;
            serde_json::from_value(value).map_err(EnvelopeError::Payload)
        }
        Value::Object(obj) if obj.contains_key("wire_version") => {
            Ok(WireEnvelope::<IpcMessage>::from_json_value(value)?.into_payload())
        }
        Value::Object(_) => {
            WireCompatibility::accept(WireCompatibility::LEGACY_UNVERSIONED)?;
            serde_json::from_value(value).map_err(EnvelopeError::Payload)
        }
        _ => Err(EnvelopeError::NotAnObject),
    }
}

fn parse_wire_version(value: &Value) -> Result<u32, EnvelopeError> {
    match value {
        Value::Number(n) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| EnvelopeError::InvalidVersion(n.to_string())),
        other => Err(EnvelopeError::InvalidVersion(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SpikeBatch, SpikeEvent};

    fn sample_spikes() -> IpcMessage {
        IpcMessage::Spikes(SpikeBatch {
            session_id: Some("sess-1".into()),
            batch_id: 7,
            timestamp: 1_700_000_000,
            spikes: vec![SpikeEvent {
                channel: 3,
                time: 11,
                strength: 0.5,
            }],
            metadata: None,
        })
    }

    fn envelope_json(version: u32, payload: Value) -> Value {
        serde_json::json!({
            "wire_version": version,
            "payload": payload,
        })
    }

    fn sample_spikes_payload() -> Value {
        serde_json::json!({
            "Spikes": {
                "session_id": "sess-1",
                "batch_id": 7,
                "timestamp": 1_700_000_000,
                "spikes": [{
                    "channel": 3,
                    "time": 11,
                    "strength": 0.5
                }],
                "metadata": null
            }
        })
    }

    #[test]
    fn compatibility_window_is_internally_consistent() {
        const {
            assert!(WireCompatibility::MIN_SUPPORTED > 0);
            assert!(WireCompatibility::CURRENT >= WireCompatibility::MIN_SUPPORTED);
            assert!(
                WireCompatibility::LEGACY_UNVERSIONED >= WireCompatibility::MIN_SUPPORTED
                    && WireCompatibility::LEGACY_UNVERSIONED <= WireCompatibility::CURRENT
            );
        }
    }

    #[test]
    fn compatibility_table_covers_min_minus_one_min_current_and_current_plus_one() {
        let min = WireCompatibility::MIN_SUPPORTED;
        let current = WireCompatibility::CURRENT;
        let rows = [
            (min - 1, Compatibility::TooOld),
            (min, Compatibility::Supported),
            (current, Compatibility::Supported),
            (current + 1, Compatibility::TooNew),
        ];
        for (version, expected) in rows {
            assert_eq!(
                classify_wire_version(version),
                expected,
                "version {version} should be {expected:?}"
            );
        }
    }

    #[test]
    fn compatibility_accept_returns_typed_too_old_and_too_new_errors() {
        let min = WireCompatibility::MIN_SUPPORTED;
        let current = WireCompatibility::CURRENT;

        let too_old = accept_wire_version(min - 1).expect_err("min-1 must fail");
        assert_eq!(
            too_old,
            CompatibilityError::TooOld {
                found: min - 1,
                min,
            }
        );
        let too_old_text = too_old.to_string();
        assert!(
            too_old_text.contains("too old"),
            "too-old error must be actionable: {too_old_text}"
        );
        assert!(
            too_old_text.contains("upgrade the producer"),
            "too-old error must say what to do: {too_old_text}"
        );

        let too_new = accept_wire_version(current + 1).expect_err("current+1 must fail");
        assert_eq!(
            too_new,
            CompatibilityError::TooNew {
                found: current + 1,
                current,
            }
        );
        let too_new_text = too_new.to_string();
        assert!(
            too_new_text.contains("too new"),
            "too-new error must be actionable: {too_new_text}"
        );
        assert!(
            too_new_text.contains("upgrade corpus-ipc"),
            "too-new error must say what to do: {too_new_text}"
        );

        let supported = accept_wire_version(current).expect("current must be accepted");
        assert_eq!(supported.get(), current);
    }

    #[test]
    fn compatibility_envelope_json_keys_stay_stable() {
        let env = WireEnvelope::new(IpcMessage::Ping);
        let json = serde_json::to_value(&env).unwrap();
        // Externally tagged unit variants encode as a JSON string, not
        // `{"Ping":null}`. Struct variants still wrap in an object.
        assert_eq!(
            json,
            serde_json::json!({
                "wire_version": 1,
                "payload": "Ping"
            })
        );
    }

    #[test]
    fn compatibility_current_envelope_round_trips_before_payload_use() {
        let encoded = encode_ipc_message_json(&sample_spikes()).unwrap();
        let decoded = decode_ipc_message_json(&encoded).unwrap();
        assert_eq!(decoded, sample_spikes());
    }

    #[test]
    fn compatibility_legacy_unversioned_fixture_decodes() {
        let legacy = sample_spikes_payload();
        let decoded = decode_ipc_message_value(legacy).unwrap();
        assert_eq!(decoded, sample_spikes());
    }

    #[test]
    fn compatibility_legacy_unit_variant_string_decodes() {
        let decoded = decode_ipc_message_json(br#""Ping""#).unwrap();
        assert_eq!(decoded, IpcMessage::Ping);
        let encoded = encode_ipc_message_json(&IpcMessage::Ping).unwrap();
        assert_eq!(decode_ipc_message_json(&encoded).unwrap(), IpcMessage::Ping);
    }

    #[test]
    fn compatibility_min_supported_envelope_decodes() {
        let json = envelope_json(WireCompatibility::MIN_SUPPORTED, sample_spikes_payload());
        let decoded = decode_ipc_message_value(json).unwrap();
        assert_eq!(decoded, sample_spikes());
    }

    #[test]
    fn compatibility_too_old_envelope_fails_before_payload_use() {
        let json = envelope_json(
            WireCompatibility::MIN_SUPPORTED - 1,
            serde_json::json!({ "Ping": null }),
        );
        let err = decode_ipc_message_value(json).expect_err("too-old must fail");
        match err {
            EnvelopeError::Compatibility(CompatibilityError::TooOld { found, min }) => {
                assert_eq!(found, WireCompatibility::MIN_SUPPORTED - 1);
                assert_eq!(min, WireCompatibility::MIN_SUPPORTED);
            }
            other => panic!("expected TooOld, got {other}"),
        }
    }

    #[test]
    fn compatibility_too_new_envelope_fails_before_unknown_payload_is_used() {
        // A future producer might send an unknown variant. The version check
        // must reject this as TooNew rather than turning it into Ping/Default
        // or a payload-unknown-variant error.
        let json = envelope_json(
            WireCompatibility::CURRENT + 1,
            serde_json::json!({ "BrandNewFutureMessage": { "x": 1 } }),
        );
        let err = decode_ipc_message_value(json).expect_err("too-new must fail");
        match err {
            EnvelopeError::Compatibility(CompatibilityError::TooNew { found, current }) => {
                assert_eq!(found, WireCompatibility::CURRENT + 1);
                assert_eq!(current, WireCompatibility::CURRENT);
            }
            other => panic!("expected TooNew before payload use, got {other}"),
        }
    }

    #[test]
    fn compatibility_too_new_missing_payload_still_fails_on_version() {
        let json = serde_json::json!({ "wire_version": WireCompatibility::CURRENT + 1 });
        let err = WireEnvelope::<IpcMessage>::from_json_value(json)
            .expect_err("too-new must fail even without payload");
        assert!(matches!(
            err,
            EnvelopeError::Compatibility(CompatibilityError::TooNew { .. })
        ));
    }

    #[test]
    fn compatibility_unknown_fields_are_ignored_on_supported_payloads() {
        let json = serde_json::json!({
            "Spikes": {
                "session_id": "sess-1",
                "batch_id": 7,
                "timestamp": 1_700_000_000,
                "spikes": [{
                    "channel": 3,
                    "time": 11,
                    "strength": 0.5
                }],
                "metadata": null,
                "future_optional_field": 123
            }
        });
        let decoded: SpikeBatch =
            serde_json::from_value(json.get("Spikes").unwrap().clone()).unwrap();
        assert_eq!(decoded.batch_id, 7);
        let via_entry = decode_ipc_message_value(json).unwrap();
        assert_eq!(via_entry, sample_spikes());
    }

    #[test]
    fn compatibility_unknown_envelope_fields_are_ignored() {
        let mut json = envelope_json(
            WireCompatibility::CURRENT,
            serde_json::json!({ "Ping": null }),
        );
        json.as_object_mut()
            .unwrap()
            .insert("future_envelope_field".into(), serde_json::json!("ok"));
        let decoded = decode_ipc_message_value(json).unwrap();
        assert_eq!(decoded, IpcMessage::Ping);
    }

    #[test]
    fn compatibility_unknown_variant_never_becomes_a_valid_default() {
        let json = serde_json::json!({ "NotARealVariant": null });
        assert!(
            serde_json::from_value::<IpcMessage>(json.clone()).is_err(),
            "unknown IpcMessage variants must fail closed"
        );
        let err = decode_ipc_message_value(json).expect_err("unknown variant must fail");
        assert!(
            matches!(err, EnvelopeError::Payload(_)),
            "legacy unknown variant is a payload error, not a default: {err}"
        );

        let enveloped = envelope_json(
            WireCompatibility::CURRENT,
            serde_json::json!({ "NotARealVariant": null }),
        );
        let err = decode_ipc_message_value(enveloped).expect_err("unknown variant must fail");
        assert!(
            matches!(err, EnvelopeError::Payload(_)),
            "supported-version unknown variant is a payload error: {err}"
        );
    }
}
