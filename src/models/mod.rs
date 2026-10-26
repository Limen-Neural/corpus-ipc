// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data types that flow over the compute backend IPC wire.
//!
//! # Transport types vs training types
//!
//! [`SpikeBatch`] and [`TraceBatch`] in this crate are **IPC transport / wire
//! payloads**. They are owned by `corpus-ipc` and serialized on the hybrid-flow
//! message bus (`IpcMessage::Spikes`, `IpcMessage::EligibilityTraces`).
//!
//! They are **not** in-memory training batches (spike-train arrays with
//! optional `times` / `targets`, or unstructured e-prop `traces`) that other
//! libraries may also call `SpikeBatch` / `TraceBatch`:
//!
//! | | `corpus-ipc` (this crate) |
//! |---|---------------------------|
//! | Domain | IPC wire / session routing |
//! | `SpikeBatch` | `session_id`, `batch_id`, timestamped [`SpikeEvent`]s |
//! | `TraceBatch` | `session_id`, `batch_id`, typed [`TraceData`] rows |
//!
//! Keep the Rust names `SpikeBatch` / `TraceBatch` so serde identifiers and
//! existing imports stay stable. Prefer [`IpcSpikeBatch`] / [`IpcTraceBatch`]
//! when the name collision would otherwise be unclear. Do not merge the IPC
//! and training definitions.

mod config;
mod de;
mod embeddings;
mod gradients;
mod message;
mod metadata;
mod neuromod;
mod spikes;
mod stimulus;
mod traces;

pub use config::{ConfigPayload, ConfigValue};
pub use embeddings::EmbeddingBatch;
pub use gradients::{GradientBatch, GradientUpdate};
pub use message::IpcMessage;
pub use metadata::BatchMetadata;
pub use neuromod::NeuromodulatorSnapshot;
pub use spikes::{IpcSpikeBatch, SpikeBatch, SpikeEvent};
pub use stimulus::StimulusBatch;
pub use traces::{IpcTraceBatch, TraceBatch, TraceData};

#[cfg(test)]
mod compat;
#[cfg(test)]
mod limits_tests;
