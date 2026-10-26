// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical IPC backend traits and the `BackendType` factory selector.
//!
//! [`IpcBackend`] is the required contract for every backend. [`HybridFlowBackend`]
//! is an optional companion for message-oriented transports that exchange
//! structured [`SpikeBatch`] / [`EmbeddingBatch`] / [`GradientBatch`] /
//! [`TraceBatch`] payloads.
//!
//! Pre-rename names (`RuntimeBackend`, `BackendType::ZmqRuntime`) remain as
//! deprecated compatibility aliases so downstream crates can migrate off the
//! #20 breaking rename without a hard cut.

use crate::{BackendError, EmbeddingBatch, GradientBatch, RustBackend, SpikeBatch, TraceBatch};

/// Unified interface for compute processing backends.
///
/// Abstracts over the compute processing layer, allowing different backend
/// implementations (Rust-native, native or network IPC) to be used
/// interchangeably. This is the common contract: every backend
/// (including `ZmqIpcBackend` when the `zmq` feature is enabled)
/// implements this trait. `ZmqIpcBackend` is `Sync` via mutex-serialized
/// socket ownership, not by marking the libzmq socket itself `Sync`.
///
/// # Output contract
///
/// `process_batch` returns a `Vec<f32>` containing the processed outputs.
/// The exact number of elements depends on the backend implementation.
pub trait IpcBackend: Send + Sync {
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

/// Deprecated compatibility name for [`IpcBackend`].
///
/// Prefer [`IpcBackend`] in new code. This alias exists so crates that still
/// import `RuntimeBackend` after the #20 rename keep compiling during the
/// deprecation window.
#[deprecated(since = "0.1.0", note = "renamed to IpcBackend")]
pub use IpcBackend as RuntimeBackend;

/// Optional high-level hybrid flow interface for message-oriented IPC backends.
///
/// Backends that support structured spike/embedding exchange can implement this
/// trait in addition to [`IpcBackend`]. It is optional; most simple backends
/// only need [`IpcBackend`]. The trait stays domain-neutral: it has no
/// service-specific methods, endpoint constants, or channel widths.
///
/// This trait does not own the wire encoding. Implementors that speak JSON
/// should decode inbound bytes with [`crate::decode_ipc_message_json`] so
/// schema versions are classified before a payload is used, then apply
/// [`crate::Validate`].
///
/// # Message coverage
///
/// | Method | `IpcMessage` variant |
/// | --- | --- |
/// | [`Self::send_spikes`] | `Spikes` |
/// | [`Self::send_embeddings`] | `Embeddings` |
/// | [`Self::try_receive_gradients`] | `GradientUpdate` |
/// | [`Self::try_receive_traces`] | `EligibilityTraces` |
///
/// Stimuli, neuromodulators, loss, config, and control variants are real
/// [`crate::IpcMessage`] values and travel on a generic structured transport.
/// They are not methods of this trait. A receive call that observes one of
/// them returns [`BackendError::Protocol`] with
/// [`crate::ProtocolError::UnexpectedVariant`] instead of discarding the frame.
///
/// With the `zmq` feature, `ZmqHybridFlowBackend` is the in-repository
/// adapter. It delegates framing to `ZmqMessagePublisher` /
/// `ZmqMessageSubscriber` (one canonical JSON envelope per ZeroMQ message).
/// `ZmqIpcBackend`, the binary readout subscriber, does not implement this trait.
///
/// # Errors
///
/// `Ok(None)` from a receive method means no inbound message is available
/// right now. [`BackendError::Protocol`] means a different variant arrived and
/// was consumed. [`BackendError::InvalidInput`] means the payload failed
/// validation. [`BackendError::CommunicationError`] covers framing, version,
/// transport failures, and structural payload decode failures.
pub trait HybridFlowBackend: Send + Sync {
    /// Send a spike batch over the transport.
    fn send_spikes(&mut self, spikes: SpikeBatch) -> Result<(), BackendError>;

    /// Send an embedding batch over the transport.
    fn send_embeddings(&mut self, embeddings: EmbeddingBatch) -> Result<(), BackendError>;

    /// Try receiving a gradient batch without blocking.
    ///
    /// Returns `Ok(Some(batch))` when a `GradientUpdate` is available,
    /// `Ok(None)` when the channel is empty, [`BackendError::Protocol`] when
    /// a different variant arrived, or another [`BackendError`] on transport
    /// or validation failure.
    fn try_receive_gradients(&mut self) -> Result<Option<GradientBatch>, BackendError>;

    /// Try receiving an eligibility trace batch without blocking.
    ///
    /// Same outcomes as [`Self::try_receive_gradients`], accepting only
    /// `EligibilityTraces`.
    fn try_receive_traces(&mut self) -> Result<Option<TraceBatch>, BackendError>;
}

/// Backend implementation selector.
///
/// Used with `BackendFactory::create` to choose the concrete implementation
/// at runtime. `Rust` is always available; `ZmqIpc` requires the `zmq` feature.
#[derive(Debug, Clone, Copy, Default)]
pub enum BackendType {
    /// Pure-Rust native backend (always available, no external deps).
    #[default]
    Rust,
    /// IPC backend via ZMQ SUB socket (requires feature `zmq`).
    #[cfg(feature = "zmq")]
    ZmqIpc,
    /// Deprecated compatibility selector for [`BackendType::ZmqIpc`].
    #[cfg(feature = "zmq")]
    #[deprecated(since = "0.1.0", note = "renamed to BackendType::ZmqIpc")]
    ZmqRuntime,
}

/// Factory for creating `IpcBackend` instances.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a boxed backend of the requested `BackendType`.
    ///
    /// The returned value implements `IpcBackend`. For `ZmqIpc`, the
    /// crate must be compiled with the `zmq` feature or this will panic at
    /// construction time (compile-time cfg guards the variant).
    ///
    /// Callers must still invoke `initialize` before `process_batch`.
    pub fn create(backend_type: BackendType) -> Box<dyn IpcBackend> {
        match backend_type {
            BackendType::Rust => Box::new(RustBackend::new()),
            #[cfg(feature = "zmq")]
            #[allow(deprecated)]
            BackendType::ZmqIpc | BackendType::ZmqRuntime => Box::new(crate::ZmqIpcBackend::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RustBackend;

    fn assert_ipc_backend<T: IpcBackend>() {}

    #[test]
    fn rust_backend_implements_ipc_backend() {
        assert_ipc_backend::<RustBackend>();
        let backend = BackendFactory::create(BackendType::Rust);
        // Trait object construction is the factory contract.
        let _: Box<dyn IpcBackend> = backend;
    }

    #[test]
    #[allow(deprecated)]
    fn runtime_backend_alias_is_ipc_backend() {
        fn assert_runtime_backend<T: RuntimeBackend>() {}
        assert_runtime_backend::<RustBackend>();
    }

    #[cfg(feature = "zmq")]
    #[test]
    fn zmq_backend_implements_ipc_backend() {
        assert_ipc_backend::<crate::ZmqIpcBackend>();
        let _: Box<dyn IpcBackend> = BackendFactory::create(BackendType::ZmqIpc);
    }

    #[cfg(feature = "zmq")]
    #[test]
    #[allow(deprecated)]
    fn deprecated_zmq_runtime_selector_creates_zmq_backend() {
        let _: Box<dyn IpcBackend> = BackendFactory::create(BackendType::ZmqRuntime);
    }
}
