// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend error type.

/// Errors that may be returned by any `BackendConnector` implementation (generic equivalent of the former TraderBackend).
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),

    #[error("Processing failed: {0}")]
    ProcessingError(String),

    #[error("Model I/O error: {0}")]
    ModelError(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
