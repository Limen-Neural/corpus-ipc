// SPDX-License-Identifier: MIT OR Apache-2.0

//! One libzmq context for the whole process.
//!
//! libzmq 4.3.4 (vendored by `zeromq-src` 0.2.6) segfaults in
//! `ypipe_t<command_t, 16>::write` when several contexts run `zmq_connect`
//! at the same time. Ubuntu 26.04 GitHub runners hit that in
//! `cargo test --all-features`: each [`crate::ZmqIpcBackend`] used to build
//! its own context, and parallel tests connected those contexts together.
//! A context is thread-safe. Sockets are not, and this crate still gives
//! each socket one owner. Clones of the context share the C context via
//! `Arc` and do not terminate it while any socket or clone remains.

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
