// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, check_count, check_finite, check_opt_string,
};

use super::BatchMetadata;
use super::de::{
    de_opt_session_id, de_spikes, de_strength, prefix_path, validate_optional_metadata,
};

/// IPC transport batch of spike events from compute processing.
///
/// This is a **wire-level** payload for [`IpcMessage::Spikes`](crate::IpcMessage::Spikes). Field names
/// and layout are part of the serialized protocol and must not be changed
/// casually.
///
/// **Not** [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)'s
/// training `SpikeBatch` (`spikes` / `times` / `targets`). That type is an
/// in-memory training batch; this type is a session-correlated list of
/// [`SpikeEvent`]s. Use [`IpcSpikeBatch`] when the name collision matters.
///
/// Names stay `SpikeBatch` so existing Rust imports and serde identifiers
/// remain compatible (RM-324 / #7).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "SpikeBatchWire")]
pub struct SpikeBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Timestamp in nanoseconds (UTC or relative).
    pub timestamp: u64,
    /// Individual spike events.
    pub spikes: Vec<SpikeEvent>,
    /// Optional batch-level metadata.
    pub metadata: Option<BatchMetadata>,
}

#[derive(Deserialize)]
struct SpikeBatchWire {
    #[serde(default, deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    timestamp: u64,
    #[serde(deserialize_with = "de_spikes")]
    spikes: Vec<SpikeEvent>,
    metadata: Option<BatchMetadata>,
}

impl TryFrom<SpikeBatchWire> for SpikeBatch {
    type Error = ValidationError;

    fn try_from(wire: SpikeBatchWire) -> Result<Self, Self::Error> {
        let batch = SpikeBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            timestamp: wire.timestamp,
            spikes: wire.spikes,
            metadata: wire.metadata,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for SpikeBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("spikes", self.spikes.len(), limits.max_spike_events)?;
        for (index, spike) in self.spikes.iter().enumerate() {
            spike
                .validate_with(limits)
                .map_err(|err| prefix_path(err, &format!("spikes[{index}]")))?;
        }
        validate_optional_metadata(&self.metadata, limits, self.spikes.len())
    }
}

/// Explicit IPC-domain name for [`SpikeBatch`].
///
/// Same type and same wire format. Prefer this alias in new code that sits
/// next to SynapticDistill training types.
pub type IpcSpikeBatch = SpikeBatch;

/// Individual spike event with channel, timing, and strength.
///
/// An element of an IPC [`SpikeBatch`], not a SynapticDistill training row.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "SpikeEventWire")]
pub struct SpikeEvent {
    /// Compute channel or channel identifier.
    pub channel: u16,
    /// Spike timestamp (relative or absolute).
    pub time: u32,
    /// Spike strength or amplitude.
    pub strength: f32,
}

#[derive(Deserialize)]
struct SpikeEventWire {
    channel: u16,
    time: u32,
    #[serde(deserialize_with = "de_strength")]
    strength: f32,
}

impl TryFrom<SpikeEventWire> for SpikeEvent {
    type Error = ValidationError;

    fn try_from(wire: SpikeEventWire) -> Result<Self, Self::Error> {
        let event = SpikeEvent {
            channel: wire.channel,
            time: wire.time,
            strength: wire.strength,
        };
        event.validate()?;
        Ok(event)
    }
}

impl Validate for SpikeEvent {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_finite("strength", self.strength)
    }
}
