// SPDX-License-Identifier: MIT OR Apache-2.0

//! # corpus-ipc
//!
//! Inter-Process Communication (IPC) library for bridging Rust to external compute engines.
//!
//! Repository: <https://github.com/Limen-Neural/corpus-ipc>
//!
//! Provides a unified interface for various backends:
//!
//! - [`IpcBackend`] — required backend contract (deprecated alias: `RuntimeBackend`)
//! - [`RustBackend`] — pure-Rust native backend (no external deps, always available)
//! - `ZmqIpcBackend` — binary readout subscriber (feature `zmq`;
//!   deprecated alias: `ZmqRuntimeBackend`). `Send` + `Sync` via mutex-serialized
//!   socket ownership; the raw `zmq::Socket` is never `Sync`.
//!
//! ## Protocol surfaces
//!
//! Three integrations stay separate. None of them is a guaranteed-delivery bus:
//!
//! 1. **Structured ZeroMQ** (feature `zmq`) — `ZmqMessagePublisher` /
//!    `ZmqMessageSubscriber` carry one compact canonical JSON
//!    `WireEnvelope<IpcMessage>` frame per message. `ZmqHybridFlowBackend`
//!    adapts that pair to [`HybridFlowBackend`]. PUB/SUB is best-effort and
//!    at-most-once.
//! 2. **Binary readout** — `ZmqIpcBackend` subscribes to `[i64 tick][N × f32 LE]`
//!    packets. It does not implement [`HybridFlowBackend`].
//! 3. **REST** — `corpus_ipc_server` (feature `server`) wraps [`IpcBackend`]
//!    only. It does not speak `IpcMessage` JSON or the binary readout framing.
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
//! | `zmq` | `ZmqIpcBackend` binary readout, plus `ZmqMessagePublisher`, `ZmqMessageSubscriber`, `ZmqTransportConfig`, and `ZmqHybridFlowBackend` (vendored libzmq; needs a C++ compiler) |
//! | `server` | `corpus_ipc_server` Axum 0.8 REST binary (`axum` + `tokio`) |
//!
//! Combine `server` and `zmq` when the REST service should select the ZMQ
//! backend via `CORPUS_IPC_BACKEND_TYPE=zmq`.
//!
//! ## Wire compatibility
//!
//! [`WireCompatibility`] is the source of truth for on-wire schema versions
//! (independent of crate semver). Decode hybrid-flow JSON through
//! [`decode_ipc_message_json`] so too-old and too-new envelopes fail closed
//! before the payload is used. See [`compatibility`] for the encoding rules
//! (unknown fields ignored; unknown [`IpcMessage`] variants never default).

mod canonical_ser;
pub mod compatibility;
pub mod error;
pub mod models;
pub mod rust_backend;
#[cfg(feature = "server")]
pub mod server;
pub mod trait_def;
pub mod validation;

#[cfg(feature = "zmq")]
pub mod zmq_backend;
#[cfg(feature = "zmq")]
pub mod zmq_message;
#[cfg(feature = "zmq")]
pub mod zmq_readout;

/// Re-export the wire-schema compatibility envelope.
pub use compatibility::{
    CanonicalEncodeError, Compatibility, CompatibilityError, EnvelopeError, SupportedWireVersion,
    WireCompatibility, WireEnvelope, accept_wire_version, classify_wire_version,
    decode_ipc_message_json, decode_ipc_message_value, encode_canonical_ipc_message,
    encode_ipc_message_json,
};
/// Re-export the main error type.
pub use error::{BackendError, ProtocolError};
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
/// Re-export typed wire validation.
pub use validation::{
    ProtocolLimits, Validate, ValidationError, ValidationKind, ValidationMeasure,
};

#[cfg(feature = "zmq")]
pub use zmq_backend::ZmqIpcBackend;
#[cfg(feature = "zmq")]
#[allow(deprecated)]
pub use zmq_backend::ZmqRuntimeBackend;
#[cfg(feature = "zmq")]
pub use zmq_message::{
    FramingFault, ZmqHybridFlowBackend, ZmqMessageError, ZmqMessagePublisher, ZmqMessageSubscriber,
    ZmqTransportConfig,
};
