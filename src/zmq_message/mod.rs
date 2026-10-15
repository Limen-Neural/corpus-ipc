// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured [`IpcMessage`](crate::IpcMessage) transport over ZeroMQ PUB/SUB.
//!
//! Requires feature `zmq`. This module is one of three protocol surfaces:
//!
//! | Surface | Framing | Type |
//! | --- | --- | --- |
//! | Structured transport (this module) | one compact canonical JSON `WireEnvelope` of [`IpcMessage`](crate::IpcMessage) per ZeroMQ message | [`ZmqMessagePublisher`], [`ZmqMessageSubscriber`], [`ZmqHybridFlowBackend`] |
//! | Binary readout | `[i64 tick][N × f32 LE]` | [`crate::ZmqIpcBackend`] |
//! | REST | HTTP JSON around [`crate::IpcBackend`] | `corpus_ipc_server` (feature `server`) |
//!
//! [`crate::ZmqIpcBackend`] does not implement [`crate::HybridFlowBackend`].
//! [`ZmqHybridFlowBackend`] does not implement [`crate::IpcBackend`].
//!
//! # Topology
//!
//! The publisher **binds**. The subscriber **connects** and subscribes to the
//! full stream (empty subscription prefix). [`ZmqTransportConfig`] socket
//! options are applied before `bind` / `connect`.
//!
//! # Delivery
//!
//! PUB/SUB is **best-effort and at-most-once**. There are no acknowledgements.
//!
//! - **Slow joiner.** Messages sent before a subscriber exists, or before its
//!   subscription has propagated, are dropped by libzmq. A successful `send`
//!   does not mean a peer observed the frame. Control and config messages have
//!   the same delivery guarantee as every other variant: none.
//! - **High-water mark.** When a peer's outbound or inbound queue is full,
//!   libzmq drops further PUB frames. `send` still returns `Ok` for that drop.
//!   A configured mark of `0` is unlimited. If the socket reports `EAGAIN` on
//!   a non-blocking send, the error is [`ZmqMessageError::HighWaterMark`].
//!   Inbound frames above [`STRUCTURED_MAX_FRAME_BYTES`] are rejected by
//!   libzmq before this crate allocates them.
//! - **Teardown.** The default linger is `0`, so `close` and `Drop` discard
//!   unsent frames and return without waiting. A later `send` or `try_recv`
//!   on a closed socket returns [`ZmqMessageError::Closed`]. Linger `-1`
//!   (infinite) is accepted by config and can block `Drop` while unsent frames
//!   remain; it is not the default.
//!
//! # Concurrency
//!
//! libzmq sockets are `Send` + `!Sync`. Publisher, subscriber, and the hybrid
//! adapter store the socket in a [`Mutex`] and do not `unsafe impl Sync`.
//! Moving an owned value to another thread is supported. Socket calls take
//! the mutex; `Drop` closes through exclusive ownership of the struct.
//!
//! [`Mutex`]: std::sync::Mutex

mod hybrid;
mod transport;

pub use hybrid::ZmqHybridFlowBackend;
pub use transport::{
    FramingFault, STRUCTURED_MAX_FRAME_BYTES, ZmqMessageError, ZmqMessagePublisher,
    ZmqMessageSubscriber, ZmqTransportConfig,
};

#[cfg(test)]
mod framing_tests;
#[cfg(test)]
mod hybrid_live;
#[cfg(test)]
mod tests;
