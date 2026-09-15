// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, check_count, check_finite,
    check_string, check_unique_by,
};

use super::de::{de_session_id, de_trace_value, de_traces, prefix_path};

/// IPC transport batch of eligibility traces for credit assignment.
///
/// This is a **wire-level** payload for [`IpcMessage::EligibilityTraces`].
/// Field names and layout are part of the serialized protocol.
///
/// **Not** [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)'s
/// training `TraceBatch` (a single unstructured `traces` field). This type
/// carries `session_id` / `batch_id` plus typed [`TraceData`] rows. Use
/// [`IpcTraceBatch`] when the name collision matters.
///
/// Names stay `TraceBatch` so existing Rust imports and serde identifiers
/// remain compatible (RM-324 / #7).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "TraceBatchWire")]
pub struct TraceBatch {
    /// Session ID for routing.
    pub session_id: String,
    /// Batch ID correlation.
    pub batch_id: u64,
    /// Eligibility trace data.
    pub traces: Vec<TraceData>,
}

/// Explicit IPC-domain name for [`TraceBatch`].
///
/// Same type and same wire format. Prefer this alias in new code that sits
/// next to SynapticDistill training types.
pub type IpcTraceBatch = TraceBatch;

#[derive(Deserialize)]
struct TraceBatchWire {
    #[serde(deserialize_with = "de_session_id")]
    session_id: String,
    batch_id: u64,
    #[serde(deserialize_with = "de_traces")]
    traces: Vec<TraceData>,
}

impl TryFrom<TraceBatchWire> for TraceBatch {
    type Error = ValidationError;

    fn try_from(wire: TraceBatchWire) -> Result<Self, Self::Error> {
        let batch = TraceBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            traces: wire.traces,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for TraceBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("session_id", &self.session_id, limits)?;
        check_count("traces", self.traces.len(), limits.max_traces)?;
        let mut total = 0;
        add_to_total(&mut total, self.traces.len(), limits, "aggregate")?;
        for (index, trace) in self.traces.iter().enumerate() {
            trace
                .validate_with(limits)
                .map_err(|err| prefix_path(err, &format!("traces[{index}]")))?;
        }
        check_unique_by("traces", &self.traces, |row| row.channel_id)?;
        Ok(())
    }
}

/// Individual eligibility trace data on the IPC wire.
///
/// An element of an IPC [`TraceBatch`], not a SynapticDistill training trace.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "TraceDataWire")]
pub struct TraceData {
    /// Channel or synapse identifier.
    #[serde(alias = "neuron_id")]
    pub channel_id: u16,
    /// Trace value (decay-modulated spike history).
    pub trace_value: f32,
    /// Timestamp of last contributing spike.
    pub last_spike_time: u32,
}

#[derive(Deserialize)]
struct TraceDataWire {
    #[serde(alias = "neuron_id")]
    channel_id: u16,
    #[serde(deserialize_with = "de_trace_value")]
    trace_value: f32,
    last_spike_time: u32,
}

impl TryFrom<TraceDataWire> for TraceData {
    type Error = ValidationError;

    fn try_from(wire: TraceDataWire) -> Result<Self, Self::Error> {
        let data = TraceData {
            channel_id: wire.channel_id,
            trace_value: wire.trace_value,
            last_spike_time: wire.last_spike_time,
        };
        data.validate()?;
        Ok(data)
    }
}

impl Validate for TraceData {
    fn validate_with(&self, _limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_finite("trace_value", self.trace_value)
    }
}
