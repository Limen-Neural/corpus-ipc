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
use serde_json::value::RawValue;

use crate::IpcMessage;
use crate::validation::{Validate, ValidationError};

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

const _: () = {
    assert!(WireCompatibility::MIN_SUPPORTED > 0);
    assert!(WireCompatibility::CURRENT >= WireCompatibility::MIN_SUPPORTED);
    assert!(
        WireCompatibility::LEGACY_UNVERSIONED >= WireCompatibility::MIN_SUPPORTED
            && WireCompatibility::LEGACY_UNVERSIONED <= WireCompatibility::CURRENT
    );
};

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
    /// Failed to parse or serialize envelope JSON.
    #[error("envelope JSON error: {0}")]
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
/// {"wire_version":1,"payload":"Ping"}
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
    ///
    /// The payload is kept as raw JSON until the version is accepted, so a
    /// too-new/too-old envelope is rejected without building `T`.
    pub fn decode_json(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        envelope_from_raw_parts(parse_raw_envelope(bytes)?)
    }

    /// Decode an already-parsed JSON value, checking version before `T`.
    ///
    /// Prefer [`Self::decode_json`] for byte input so the payload is not first
    /// materialized as a [`Value`] tree. This entry point exists for callers
    /// that already hold a `Value`.
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

/// Fail-closed errors from [`encode_canonical_ipc_message`].
///
/// This is a **dedicated** encode error, deliberately separate from
/// [`EnvelopeError`]: its match arms are part of a different contract and must
/// not be folded into the decode error. Canonical encoding fails before any
/// bytes are produced, either because the payload did not pass
/// [`Validate::validate`] (for example a non-finite `f32`, which JSON would
/// otherwise emit as a silent `null`) or because serialization itself failed.
#[derive(Debug, thiserror::Error)]
pub enum CanonicalEncodeError {
    /// The payload failed [`Validate::validate`] before serialization.
    ///
    /// Non-finite floats (`NaN`, `±inf`) are reported here as
    /// [`ValidationKind::NonFinite`](crate::ValidationKind::NonFinite) rather
    /// than serialized as `null`.
    #[error("payload failed validation before canonical encoding: {0}")]
    Validation(#[from] ValidationError),
    /// Serialization of the validated envelope failed.
    #[error("canonical JSON serialization error: {0}")]
    Json(serde_json::Error),
}

/// Encode `message` as **canonical** wire-version-1 envelope bytes.
///
/// This is the single canonical encoder for the project wire profile
/// (see `docs/wire-encoding.md`). It is *not* RFC 8785 / JCS; it is this
/// crate's own profile. Guarantees:
///
/// 1. **Validated first.** [`Validate::validate`] runs before serialization,
///    so a directly constructed non-finite `f32` is rejected with a typed
///    [`CanonicalEncodeError::Validation`] instead of being emitted as JSON
///    `null`.
/// 2. **Deterministic bytes.** Object keys are emitted in sorted (byte-wise)
///    order at every nesting level, independent of `HashMap` insertion order
///    or per-process hash seed. The same message therefore encodes to
///    identical bytes across runs and processes. Determinism is achieved by
///    routing through [`serde_json::Value`] and calling
///    [`serde_json::Value::sort_all_objects`], which recursively sorts every
///    nested object. This does not rely on the default `BTreeMap` backing:
///    if a consumer's dependency graph enables serde_json's `preserve_order`
///    feature (unified across the graph by Cargo), `Value` uses an
///    insertion-ordered `IndexMap` and the explicit sort still applies. The
///    public [`crate::ConfigPayload`] / [`crate::BatchMetadata`] field types
///    stay `HashMap`.
/// 3. **Wire version 1.** The payload is wrapped in [`WireEnvelope::new`], so
///    `wire_version` is [`WireCompatibility::CURRENT`].
///
/// Bytes produced here decode through [`decode_ipc_message_json`].
///
/// ```
/// use corpus_ipc::{decode_ipc_message_json, encode_canonical_ipc_message, IpcMessage};
///
/// let bytes = encode_canonical_ipc_message(&IpcMessage::Ping).unwrap();
/// assert_eq!(decode_ipc_message_json(&bytes).unwrap(), IpcMessage::Ping);
/// ```
pub fn encode_canonical_ipc_message(message: &IpcMessage) -> Result<Vec<u8>, CanonicalEncodeError> {
    // 1. Reuse validation as the single pre-serialization gate. This rejects
    //    non-finite floats before any bytes exist (no silent null).
    message.validate()?;
    // 2. Reuse the envelope so wire_version == CURRENT (1).
    let envelope = WireEnvelope::new(message);
    // 3. Route through Value and sort every nested object's keys. Without
    //    serde_json's `preserve_order` feature, `Value`'s map is already a
    //    sorted `BTreeMap` and `sort_all_objects` is a no-op; with it (which a
    //    consumer's dependency graph can enable via Cargo feature
    //    unification), `Value` becomes an insertion-ordered `IndexMap`, so the
    //    explicit recursive sort is what keeps the bytes independent of
    //    `HashMap` iteration order and the wider feature set.
    let mut value = serde_json::to_value(&envelope).map_err(CanonicalEncodeError::Json)?;
    value.sort_all_objects();
    serde_json::to_vec(&value).map_err(CanonicalEncodeError::Json)
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
    match parse_raw_envelope(bytes) {
        Ok(raw) if matches!(raw.wire_version, OptionalValue::Present(_)) => {
            Ok(envelope_from_raw_parts::<IpcMessage>(raw)?.into_payload())
        }
        Ok(_) => {
            WireCompatibility::accept(WireCompatibility::LEGACY_UNVERSIONED)?;
            serde_json::from_slice(bytes).map_err(EnvelopeError::Payload)
        }
        Err(EnvelopeError::NotAnObject) => {
            // Unit-variant strings (`"Ping"`) are not envelopes.
            let value = serde_json::from_slice(bytes).map_err(EnvelopeError::Json)?;
            decode_ipc_message_value(value)
        }
        Err(EnvelopeError::Json(raw_error)) => {
            let value = serde_json::from_slice::<Value>(bytes).map_err(EnvelopeError::Json)?;
            if let Value::Object(obj) = &value
                && obj.contains_key("wire_version")
            {
                return Err(EnvelopeError::Json(raw_error));
            }
            decode_ipc_message_value(value)
        }
        Err(other) => Err(other),
    }
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

/// Distinguishes a missing `wire_version` field from an explicit JSON `null`.
#[derive(Default)]
enum OptionalValue {
    #[default]
    Absent,
    Present(Value),
}

impl<'de> Deserialize<'de> for OptionalValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::Present(Value::deserialize(deserializer)?))
    }
}

/// Envelope JSON with the payload left unparsed until the version is accepted.
#[derive(Deserialize)]
struct RawEnvelope {
    #[serde(default)]
    wire_version: OptionalValue,
    #[serde(default)]
    payload: OptionalRaw,
}

/// Distinguishes a missing `payload` field from an explicit JSON `null`.
#[derive(Default)]
enum OptionalRaw {
    #[default]
    Absent,
    Present(Box<RawValue>),
}

impl<'de> Deserialize<'de> for OptionalRaw {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::Present(Box::<RawValue>::deserialize(deserializer)?))
    }
}

fn envelope_from_raw_parts<T: DeserializeOwned>(
    raw: RawEnvelope,
) -> Result<WireEnvelope<T>, EnvelopeError> {
    let OptionalValue::Present(version_val) = raw.wire_version else {
        return Err(EnvelopeError::MissingVersion);
    };
    let version = parse_wire_version(&version_val)?;
    WireCompatibility::accept(version)?;
    let OptionalRaw::Present(payload_raw) = raw.payload else {
        return Err(EnvelopeError::MissingPayload);
    };
    let payload = serde_json::from_str(payload_raw.get()).map_err(EnvelopeError::Payload)?;
    Ok(WireEnvelope {
        wire_version: version,
        payload,
    })
}

fn parse_raw_envelope(bytes: &[u8]) -> Result<RawEnvelope, EnvelopeError> {
    match bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) {
        Some(b'{') => serde_json::from_slice(bytes).map_err(EnvelopeError::Json),
        Some(_) => {
            if json_is_valid_non_object(bytes) {
                Err(EnvelopeError::NotAnObject)
            } else {
                serde_json::from_slice(bytes).map_err(EnvelopeError::Json)
            }
        }
        None => serde_json::from_slice(bytes).map_err(EnvelopeError::Json),
    }
}

fn json_is_valid_non_object(bytes: &[u8]) -> bool {
    matches!(serde_json::from_slice::<Value>(bytes), Ok(value) if !value.is_object())
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
