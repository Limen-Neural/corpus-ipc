// SPDX-License-Identifier: MIT OR Apache-2.0

//! # corpus-ipc
//!
//! Inter-Process Communication (IPC) library for bridging Rust to external compute engines.
//!
//! Provides a unified interface for various backends:
//!
//! - [`IpcBackend`] — required backend contract (deprecated alias: `RuntimeBackend`)
//! - [`RustBackend`] — pure-Rust native backend (no external deps, always available)
//! - `ZmqIpcBackend` — IPC backend via ZMQ SUB socket (feature `zmq`;
//!   deprecated alias: `ZmqRuntimeBackend`)
//!
//! ## Feature flags
//!
//! The default crate is schema and transport only: wire models, [`IpcBackend`],
//! and [`RustBackend`]. Optional stacks are opt-in so library consumers do not
//! compile or link ZeroMQ or the HTTP service:
//!
//! | Feature | Enables |
//! | --- | --- |
//! | *(none / default)* | Wire models and `RustBackend` |
//! | `zmq` | `ZmqIpcBackend` (vendored libzmq; needs a C++ compiler) |
//! | `server` | `corpus_ipc_server` Axum REST binary (`axum` + `tokio`) |
//!
//! Combine `server` and `zmq` when the REST service should select the ZMQ
//! backend via `CORPUS_IPC_BACKEND_TYPE=zmq`.

pub mod error;
pub mod models;
pub mod rust_backend;
pub mod trait_def;

#[cfg(feature = "zmq")]
pub mod zmq_backend;

/// Re-export the main error type.
pub use error::BackendError;
/// Re-export all public data models used on the wire.
pub use models::{
    BatchMetadata, ConfigPayload, ConfigValue, EmbeddingBatch, GradientBatch, GradientUpdate,
    IpcMessage, IpcSpikeBatch, IpcTraceBatch, NeuromodulatorSnapshot, SpikeBatch, SpikeEvent,
    StimulusBatch, TraceBatch, TraceData,
};
/// Re-export the core trait, factory, and backend.
pub use rust_backend::RustBackend;
#[allow(deprecated)]
pub use trait_def::{BackendFactory, BackendType, HybridFlowBackend, IpcBackend, RuntimeBackend};

#[cfg(feature = "zmq")]
pub use zmq_backend::ZmqIpcBackend;
#[cfg(feature = "zmq")]
#[allow(deprecated)]
pub use zmq_backend::ZmqRuntimeBackend;
