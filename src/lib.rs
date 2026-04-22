//! # corpus-ipc
//!
//! Inter-Process Communication (IPC) library for bridging Rust to external compute engines.
//!
//! Provides a unified interface for various backends:
//!
//! - [`RustBackend`] — pure-Rust native backend (no external deps, always available)
//! - [`ZmqBrainBackend`] — IPC backend via ZMQ SUB socket (feature `zmq`)

pub mod error;
pub mod trait_def;
pub mod rust_backend;
pub mod models;

#[cfg(feature = "zmq")]
pub mod zmq_backend;

pub use error::BackendError;
pub use trait_def::{NeuralBackend, BackendType};
pub use rust_backend::RustBackend;
pub use models::NeroManifoldSnapshot;

#[cfg(feature = "zmq")]
pub use zmq_backend::ZmqBrainBackend;
