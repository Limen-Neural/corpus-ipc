// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend error type.

/// Errors that may be returned by any `RuntimeBackend` implementation.
///
/// These cover initialization, processing, model persistence, IPC communication,
/// and invalid caller input. The inner `String` carries a human-readable
/// diagnostic (implementation-defined).
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Backend failed to initialize (e.g. model load failure, ZMQ connect error).
    #[error("Initialization failed: {0}")]
    InitializationError(String),

    /// Error during `process_batch` (e.g. shape mismatch, backend internal failure).
    #[error("Processing failed: {0}")]
    ProcessingError(String),

    /// Model save/load or filesystem I/O error.
    #[error("Model I/O error: {0}")]
    ModelError(String),

    /// IPC / transport communication error (ZMQ send/recv, framing, timeout).
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Caller supplied invalid input (e.g. empty batch where disallowed, bad config).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
