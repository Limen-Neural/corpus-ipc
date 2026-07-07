// SPDX-License-Identifier: MIT OR Apache-2.0

//! `RuntimeBackend` trait and `BackendType` enumeration.

use crate::{BackendError, EmbeddingBatch, GradientBatch, RustBackend, SpikeBatch, TraceBatch};

/// Unified interface for compute processing backends.
///
/// Abstracts over the compute processing layer, allowing different backend
/// implementations (Rust-native, Julia jlrs, ZMQ IPC) to be used
/// interchangeably.
///
/// # Output contract
///
/// `process_batch` returns a `Vec<f32>` containing the processed outputs.
/// The exact number of elements depends on the backend implementation.
pub trait RuntimeBackend: Send + Sync {
    /// Process a dynamic slice of input signals through the compute backend.
    ///
    /// # Arguments
    /// - `inputs` — A dynamically sized slice of `f32` input signals.
    fn process_batch(&mut self, inputs: &[f32]) -> Result<Vec<f32>, BackendError>;

    /// Initialise backend (load model weights, connect to IPC socket, etc.).
    ///
    /// Must be called before `process_batch`. Idempotent on success.
    fn initialize(&mut self, model_path: Option<&str>) -> Result<(), BackendError>;

    /// Persist current model state to `model_path`.
    fn save_state(&self, model_path: &str) -> Result<(), BackendError>;

    /// Return per-channel spike states (true = spiked on last tick).
    fn get_spike_states(&self) -> Vec<bool>;

    /// Reset internal network state (membrane potentials, caches).
    ///
    /// For connection-oriented backends (e.g. ZMQ), this may also drop the
    /// transport socket and clear the initialized flag, requiring a subsequent
    /// call to `initialize()` before the next `process_batch()`. Callers
    /// should treat `reset()` + `process_batch()` without re-initialization
    /// as potentially requiring re-connection for such backends.
    fn reset(&mut self) -> Result<(), BackendError>;
}

/// Optional high-level hybrid flow interface for message-oriented IPC backends.
///
/// Backends that support structured spike/embedding exchange can implement this
/// trait in addition to `RuntimeBackend`. It is optional; most simple
/// backends only need `RuntimeBackend`.
pub trait HybridFlowBackend: Send + Sync {
    /// Send a spike batch over the transport.
    fn send_spikes(&mut self, spikes: SpikeBatch) -> Result<(), BackendError>;

    /// Send an embedding batch over the transport.
    fn send_embeddings(&mut self, embeddings: EmbeddingBatch) -> Result<(), BackendError>;

    /// Try receiving a gradient batch without blocking.
    /// Returns `Ok(Some(batch))` when data is available, `Ok(None)` when the
    /// channel is empty, or `Err` on transport failure.
    fn try_receive_gradients(&mut self) -> Result<Option<GradientBatch>, BackendError>;

    /// Try receiving an eligibility trace batch without blocking.
    fn try_receive_traces(&mut self) -> Result<Option<TraceBatch>, BackendError>;
}

/// Backend implementation selector.
///
/// Used with `BackendFactory::create` to choose the concrete implementation
/// at runtime. `Rust` is always available; `ZmqRuntime` requires the `zmq` feature.
#[derive(Debug, Clone, Copy, Default)]
pub enum BackendType {
    /// Pure-Rust native backend (always available, no external deps).
    #[default]
    Rust,
    /// Julia IPC backend via ZMQ SUB socket (requires feature `zmq`).
    #[cfg(feature = "zmq")]
    ZmqRuntime,
}

/// Factory for creating `RuntimeBackend` instances.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a boxed backend of the requested `BackendType`.
    ///
    /// The returned value implements `RuntimeBackend`. For `ZmqRuntime`, the
    /// crate must be compiled with the `zmq` feature or this will panic at
    /// construction time (compile-time cfg guards the variant).
    ///
    /// Callers must still invoke `initialize` before `process_batch`.
    pub fn create(backend_type: BackendType) -> Box<dyn RuntimeBackend> {
        match backend_type {
            BackendType::Rust => Box::new(RustBackend::new()),
            #[cfg(feature = "zmq")]
            BackendType::ZmqRuntime => Box::new(crate::ZmqRuntimeBackend::new()),
        }
    }
}
