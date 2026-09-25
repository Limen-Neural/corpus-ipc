// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{ProtocolLimits, Validate, ValidationError, check_range};

/// Typed neuromodulator snapshot for JSON / explicit byte ingress.
///
/// # Relationship to ZMQ SUB readouts
///
/// The ZMQ SUB readout backend (`ZmqIpcBackend`, feature `zmq`) treats an
/// 88-byte binary frame as **tick + 20 `f32` values** and does **not**
/// auto-split bytes `[72..88]` into four
/// modulator scores. The layout below describes a **historical generic packet**
/// interpretation used when callers explicitly slice scores (e.g.
/// [`Self::from_scores`]) — not something the ZMQ subscriber infers from length
/// alone, because 88 bytes is ambiguous between 20 readouts and 16 readouts +
/// 4 modulators.
///
/// # Historical generic packet layout (bytes 72–87 when explicitly parsed)
/// ```text
/// [72..76]  dopamine       f32 LE   reward / learning-rate gate
/// [76..80]  cortisol       f32 LE   stress / inhibition
/// [80..84]  acetylcholine  f32 LE   focus / signal-to-noise
/// [84..88]  tempo          f32 LE   clock-driven timing scale
/// ```
///
/// # References
///
/// - Schultz, W. (1998). Predictive reward signal of dopamine channels.
///   *Journal of Neurophysiology*, 80(1), 1–27.
/// - Arnsten, A. F. T. (2009). Stress signalling pathways that impair
///   prefrontal cortex structure and function.
///   *Nature Reviews Neuroscience*, 10(6), 410–422.
/// - Hasselmo, M. E. (1999). Neuromodulation: acetylcholine and memory
///   consolidation. *Trends in Cognitive Sciences*, 3(9), 351–359.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NeuromodulatorSnapshotWire")]
pub struct NeuromodulatorSnapshot {
    /// Tick counter from the remote compute (monotonically increasing).
    pub tick: i64,
    /// Dopamine level (reward / STDP learning-rate gate). Range [0, 1].
    pub dopamine: f32,
    /// Cortisol level (thermal/power stress inhibition). Range [0, 1].
    pub cortisol: f32,
    /// Acetylcholine level (focus / signal-to-noise ratio). Range [0, 1].
    pub acetylcholine: f32,
    /// Tempo scale (clock-driven timing; 1.0 = nominal). Range [0.5, 2.0].
    pub tempo: f32,
}

/// Deserialization-only shadow of [`NeuromodulatorSnapshot`] with the
/// identical wire shape. `NeuromodulatorSnapshot`'s real `Deserialize` impl
/// goes through this type and [`NeuromodulatorSnapshot::validate`] so an
/// out-of-range or non-finite field is rejected at deserialization instead
/// of silently reaching consumers (this type is reachable via
/// [`IpcMessage::Neuromodulators`]).
#[derive(Deserialize)]
pub(super) struct NeuromodulatorSnapshotWire {
    pub(super) tick: i64,
    pub(super) dopamine: f32,
    pub(super) cortisol: f32,
    pub(super) acetylcholine: f32,
    pub(super) tempo: f32,
}

impl TryFrom<NeuromodulatorSnapshotWire> for NeuromodulatorSnapshot {
    type Error = ValidationError;

    fn try_from(wire: NeuromodulatorSnapshotWire) -> Result<Self, Self::Error> {
        let snapshot = NeuromodulatorSnapshot {
            tick: wire.tick,
            dopamine: wire.dopamine,
            cortisol: wire.cortisol,
            acetylcholine: wire.acetylcholine,
            tempo: wire.tempo,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

impl NeuromodulatorSnapshot {
    /// Parse from four explicit score floats (historical bytes `[72..88]` layout).
    ///
    /// Calls [`Validate::validate`] internally and returns `Err` if the decoded
    /// bytes are out of the documented ranges or non-finite. This keeps
    /// byte-packet ingress consistent with JSON ingress (`IpcMessage::Neuromodulators`
    /// deserialization also validates) — otherwise a bad packet would only
    /// surface as a confusing "failed to deserialize" error at the receiver,
    /// pointing away from where the bad bytes actually came from.
    pub fn from_scores(tick: i64, scores: &[f32; 4]) -> Result<Self, ValidationError> {
        let snapshot = Self {
            tick,
            dopamine: scores[0],
            cortisol: scores[1],
            acetylcholine: scores[2],
            tempo: scores[3],
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

impl Validate for NeuromodulatorSnapshot {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_range("dopamine", self.dopamine, 0.0, 1.0)?;
        check_range("cortisol", self.cortisol, 0.0, 1.0)?;
        check_range("acetylcholine", self.acetylcholine, 0.0, 1.0)?;
        check_range("tempo", self.tempo, 0.5, 2.0)?;
        Ok(())
    }
}
