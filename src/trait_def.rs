//! `NeuralBackend` trait and `BackendType` enumeration.

use crate::{BackendError, RustBackend};

/// Unified interface for neural processing backends.
///
/// Abstracts over the neural processing layer, allowing different backend
/// implementations (Rust-native, Julia jlrs, ZMQ IPC) to be used
/// interchangeably.
///
/// # Output contract
///
/// `process_signals` returns a `Vec<f32>` containing the processed outputs.
/// The exact number of elements depends on the backend implementation.
pub trait NeuralBackend: Send + Sync {
    /// Process a dynamic slice of input signals through the neural backend.
    ///
    /// # Arguments
    /// - `inputs` — A dynamically sized slice of `f32` input signals.
    fn process_signals(
        &mut self,
        inputs: &[f32],
    ) -> Result<Vec<f32>, BackendError>;

    /// Initialise backend (load model weights, connect to IPC socket, etc.).
    ///
    /// Must be called before `process_signals`. Idempotent on success.
    fn initialize(&mut self, model_path: Option<&str>) -> Result<(), BackendError>;

    /// Persist current model state to `model_path`.
    fn save_state(&self, model_path: &str) -> Result<(), BackendError>;

    /// Return per-neuron spike states (true = spiked on last tick).
    fn get_spike_states(&self) -> Vec<bool>;

    /// Reset internal network state (membrane potentials, caches).
    fn reset(&mut self) -> Result<(), BackendError>;
}

/// Backend implementation selector.
#[derive(Debug, Clone, Copy, Default)]
pub enum BackendType {
    /// Pure-Rust native backend (always available, no external deps).
    #[default]
    Rust,
    /// Julia IPC backend via ZMQ SUB socket (requires feature `zmq`).
    #[cfg(feature = "zmq")]
    ZmqBrain,
}

/// Factory for creating `NeuralBackend` instances.
pub struct BackendFactory;

impl BackendFactory {
    pub fn create(backend_type: BackendType) -> Box<dyn NeuralBackend> {
        match backend_type {
            BackendType::Rust => Box::new(RustBackend::new()),
            #[cfg(feature = "zmq")]
            BackendType::ZmqBrain => Box::new(crate::ZmqBrainBackend::new()),
        }
    }
}
