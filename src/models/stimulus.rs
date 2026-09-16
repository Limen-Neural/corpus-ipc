// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, check_count, check_finite_slice, check_opt_string,
};

use super::BatchMetadata;
use super::de::{de_opt_session_id, de_valid_mask, de_values, validate_optional_metadata};

/// Canonical wire batch of continuous, domain-neutral runtime stimulus values.
///
/// This is the typed replacement for downstream services' ad-hoc stimulus
/// payloads — e.g. `thalamic-relay`'s untagged `{"type":"Stimuli","values":[...]}`
/// UDP JSON, and `brainstem-daemon`'s local `IngressPacket { stimuli, modulators }`
/// struct. `corpus-ipc` owns this schema; downstream services should decode/encode
/// through [`IpcMessage::Stimuli`] instead of a private struct or raw
/// `serde_json::Value` field indexing.
///
/// # Channel width
///
/// `values.len()` is the channel count. It is **not fixed** by this crate —
/// do not encode any particular network's input width (e.g. Spikenaut's
/// current axon count) into this type. Consumers determine width at runtime
/// from the batch itself.
///
/// # Invalid / missing channel semantics
///
/// `valid_mask`, when `Some`, must be the same length as `values`.
/// `valid_mask[i] == false` means channel `i` has no valid reading this tick;
/// the corresponding `values[i]` is a placeholder (`0.0` by convention) and
/// must **not** be interpreted as a real zero-valued reading. When
/// `valid_mask` is `None`, every entry in `values` is valid.
///
/// This is a deliberate improvement over ad-hoc formats (e.g. `thalamic-relay`'s
/// UDP handler) that silently coerce missing or non-numeric channels to `0.0`
/// with no way to distinguish "sensor read zero" from "no data this tick."
///
/// Fields are public (matching this crate's other wire batches, e.g.
/// [`SpikeBatch`]) for direct Rust construction, but **deserialization
/// enforces the `valid_mask`-length invariant**: a JSON/wire payload with a
/// `valid_mask` whose length differs from `values.len()` fails to
/// deserialize (via [`StimulusBatch::validate`] through a `TryFrom` shadow
/// type), rather than silently producing an inconsistent instance. Call
/// [`Validate::validate`] explicitly after constructing one directly in
/// Rust (which bypasses deserialization) to get the same check.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "StimulusBatchWire")]
pub struct StimulusBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Timestamp in nanoseconds (UTC or relative).
    pub timestamp: u64,
    /// Continuous stimulus values. Length is the channel count; not fixed by this crate.
    pub values: Vec<f32>,
    /// Optional per-channel validity mask, same length as `values` when present.
    /// `false` marks a channel as invalid/missing for this tick (see type docs).
    pub valid_mask: Option<Vec<bool>>,
    /// Optional batch-level metadata.
    pub metadata: Option<BatchMetadata>,
}

impl Validate for StimulusBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("values", self.values.len(), limits.max_channel_values)?;
        check_finite_slice("values", &self.values)?;
        if let Some(mask) = &self.valid_mask {
            check_count("valid_mask", mask.len(), limits.max_channel_values)?;
            if mask.len() != self.values.len() {
                return Err(ValidationError::length_mismatch(
                    "valid_mask",
                    mask.len(),
                    self.values.len(),
                ));
            }
        }
        validate_optional_metadata(&self.metadata, limits, self.values.len())
    }
}

/// Deserialization-only shadow of [`StimulusBatch`] with the identical wire
/// shape. `StimulusBatch`'s real `Deserialize` impl goes through this type and
/// [`Validate::validate`] so a malformed `valid_mask` length, non-finite
/// value, or oversize payload is rejected at deserialization.
#[derive(Deserialize)]
struct StimulusBatchWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    timestamp: u64,
    #[serde(deserialize_with = "de_values")]
    values: Vec<f32>,
    #[serde(deserialize_with = "de_valid_mask")]
    valid_mask: Option<Vec<bool>>,
    metadata: Option<BatchMetadata>,
}

impl TryFrom<StimulusBatchWire> for StimulusBatch {
    type Error = ValidationError;

    fn try_from(wire: StimulusBatchWire) -> Result<Self, Self::Error> {
        let batch = StimulusBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            timestamp: wire.timestamp,
            values: wire.values,
            valid_mask: wire.valid_mask,
            metadata: wire.metadata,
        };
        batch.validate()?;
        Ok(batch)
    }
}
