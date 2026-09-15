// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{ProtocolLimits, Validate, ValidationError, check_finite};

use super::de::de_loss;
use super::{
    ConfigPayload, EmbeddingBatch, GradientBatch, NeuromodulatorSnapshot, SpikeBatch,
    StimulusBatch, TraceBatch,
};

/// Core message enum for cross-process IPC.
///
/// Messages are separated into:
/// - Input messages (spikes, embeddings, stimuli, neuromodulators, config)
/// - Output messages (gradients, traces, training status)
/// - Control messages (shutdown, ping)
///
/// Variant names (`Spikes`, `EligibilityTraces`, …) are serde identifiers and
/// must stay stable. The payloads they wrap are IPC transport types, not
/// SynapticDistill training structs (see the module docs).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum IpcMessage {
    // Input messages
    /// Wire envelope for an IPC [`SpikeBatch`] (not a SynapticDistill training batch).
    Spikes(SpikeBatch),
    Embeddings(EmbeddingBatch),
    /// Typed continuous runtime stimulus ingress (e.g. `thalamic-relay` ->
    /// `brainstem-daemon`). See [`StimulusBatch`] for channel-width and
    /// invalid/missing-channel semantics.
    Stimuli(StimulusBatch),
    /// Typed neuromodulator ingress, replacing an unstructured float tail.
    /// See [`NeuromodulatorSnapshot`] and [`Validate`].
    Neuromodulators(NeuromodulatorSnapshot),
    Loss(#[serde(deserialize_with = "de_loss")] f32),
    ConfigUpdate(ConfigPayload),

    // Output messages
    GradientUpdate(GradientBatch),
    /// Wire envelope for an IPC [`TraceBatch`] (not a SynapticDistill training batch).
    EligibilityTraces(TraceBatch),
    TrainingComplete,

    // Control
    Shutdown,
    Ping,
}

impl Validate for IpcMessage {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        match self {
            Self::Spikes(batch) => batch.validate_with(limits),
            Self::Embeddings(batch) => batch.validate_with(limits),
            Self::Stimuli(batch) => batch.validate_with(limits),
            Self::Neuromodulators(snapshot) => snapshot.validate_with(limits),
            Self::ConfigUpdate(payload) => payload.validate_with(limits),
            Self::GradientUpdate(batch) => batch.validate_with(limits),
            Self::EligibilityTraces(batch) => batch.validate_with(limits),
            Self::Loss(value) => check_finite("Loss", *value),
            Self::TrainingComplete | Self::Shutdown | Self::Ping => Ok(()),
        }
    }
}
