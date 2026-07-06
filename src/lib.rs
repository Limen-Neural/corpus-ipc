// SPDX-License-Identifier: MIT OR Apache-2.0

//! # corpus-ipc
//!
//! Inter-Process Communication (IPC) library for bridging Rust to external compute engines.
//!
//! Provides a unified interface for various backends:
//!
//! - [`RustBackend`] — pure-Rust native backend (no external deps, always available)
//! - [`ZmqRuntimeBackend`] — IPC backend via ZMQ SUB socket (feature `zmq`)

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
    RuntimeSnapshot, SpikeBatch, SpikeEvent, RuntimeMessage, TraceBatch, TraceData,
};
/// Re-export the core trait and factory.
pub use rust_backend::RustBackend;
pub use trait_def::{RuntimeBackend, BackendType, HybridFlowBackend};


#[cfg(feature = "zmq")]
pub use zmq_backend::ZmqRuntimeBackend;
