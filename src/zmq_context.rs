// SPDX-License-Identifier: MIT OR Apache-2.0

//! One libzmq context for the whole process.
//!
//! A context is thread-safe. Sockets are not, and this crate still gives
//! each socket one owner. Clones share the C context via `Arc`. The
//! `OnceLock` keeps one clone alive until process exit, so dropping a
//! backend closes its sockets without terminating the context under
//! another thread that is still using it.

use std::sync::OnceLock;

static CONTEXT: OnceLock<zmq::Context> = OnceLock::new();

/// Clone of the process-wide ZeroMQ context.
///
/// The first call creates the context. Later calls clone that handle.
/// The `OnceLock` keeps one clone alive until process exit, so dropping a
/// backend or a publisher closes its sockets without tearing the context
/// down underneath another thread that is still connecting.
pub(crate) fn shared_context() -> zmq::Context {
    CONTEXT.get_or_init(zmq::Context::new).clone()
}
