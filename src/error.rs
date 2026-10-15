// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend error type.

/// Errors returned by [`crate::IpcBackend`] and [`crate::HybridFlowBackend`].
///
/// These cover initialization, processing, model persistence, IPC communication,
/// invalid caller input, and structured-message protocol rejection. Most
/// variants carry a human-readable `String`. [`BackendError::Protocol`] carries
/// a [`ProtocolError`] so a hybrid-flow receiver can reject an unexpected
/// [`crate::IpcMessage`] variant without overloading a transport or validation
/// failure.
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

    /// A structured message was valid on the wire but not acceptable for this call.
    ///
    /// Hybrid-flow receivers return this when an inbound [`crate::IpcMessage`]
    /// decodes and validates, then turns out to be a different variant than the
    /// method accepts. The frame is already consumed; it is not dropped silently
    /// and it is not retried.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

/// Structured-message protocol rejection.
///
/// This is distinct from transport failure ([`BackendError::CommunicationError`])
/// and from payload validation ([`BackendError::InvalidInput`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    /// The decoded message variant is not the one this call accepts.
    #[error("unexpected IpcMessage variant `{found}`; expected {expected}")]
    UnexpectedVariant {
        /// Serde variant name that arrived (`Ping`, `Spikes`, …).
        found: &'static str,
        /// Serde variant name this call accepts.
        expected: &'static str,
    },
}
