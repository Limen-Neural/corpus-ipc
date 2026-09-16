// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data types that flow over the compute backend IPC wire.
//!
//! # Transport types vs training types
//!
//! [`SpikeBatch`] and [`TraceBatch`] in this crate are **IPC transport / wire
//! payloads**. They are owned by `corpus-ipc` and serialized on the hybrid-flow
//! message bus (`IpcMessage::Spikes`, `IpcMessage::EligibilityTraces`).
//!
//! They are **not** the same types as `SpikeBatch` / `TraceBatch` in
//! [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)
//! (`src/types.jl`). Those are **training-facing** in-memory batches:
//!
//! | | `corpus-ipc` (this crate) | `SynapticDistill.jl` |
//! |---|---------------------------|----------------------|
//! | Domain | IPC wire / session routing | SNN training step |
//! | `SpikeBatch` | `session_id`, `batch_id`, timestamped [`SpikeEvent`]s | spike trains (`spikes`), optional `times` / `targets` |
//! | `TraceBatch` | `session_id`, `batch_id`, typed [`TraceData`] rows | unstructured `traces` for e-prop / credit assignment |
//!
//! Keep the Rust names `SpikeBatch` / `TraceBatch` so serde identifiers and
//! existing imports stay stable. Prefer [`IpcSpikeBatch`] / [`IpcTraceBatch`]
//! when the cross-repo collision would otherwise be unclear. Do not merge the
//! IPC and training definitions.

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
