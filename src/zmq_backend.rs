// SPDX-License-Identifier: MIT OR Apache-2.0

//! ZMQ SUB backend — reads compute data packets from the remote compute IPC socket.
//!
//! Requires feature `zmq`.
//!
//! # Concurrency
//!
//! libzmq sockets are not thread-safe ([`zmq(7)`](http://api.zeromq.org/4-3:zmq)
//! *Thread safety*). The `zmq` 0.10 crate therefore implements `Send` for
//! [`zmq::Socket`] and leaves it `!Sync`. This backend never re-asserts `Sync`
//! on the raw socket. Instead:
//!
//! - **Move (`Send`)**: [`ZmqIpcBackend`] (and its SUB socket) may be moved to
//!   another thread. libzmq allows that migration across a memory barrier;
//!   transferring ownership of this struct provides one.
//! - **Share (`Sync`)**: the public type is `Sync` only because the socket is
//!   stored in a [`Mutex`]. `zmq::Socket` methods are `&self`, which is
//!   interior mutability at the C layer — they must not run concurrently.
//!   Recv and `reset`'s close path take the mutex. Create/connect run on the
//!   initializing thread before the socket is stored. Drop of the backend
//!   closes via exclusive ownership of the struct.
//! - **`&mut self` cache boundary**: `initialize`, `process_batch`, and `reset`
//!   still take `&mut self` for the readout cache and initialized flag.
//!   Shared `&ZmqIpcBackend` may call [`IpcBackend::save_state`] and
//!   [`IpcBackend::get_spike_states`] concurrently with each other, but not
//!   with a `&mut self` call (ordinary Rust aliasing). The REST server wraps
//!   a backend in `Arc<Mutex<Box<dyn IpcBackend>>>`.
//! - **Initialize**: creates, subscribes, and connects the SUB socket on the
//!   calling thread, then stores it under the mutex. Idempotent after success.
//! - **Process**: non-blocking `recv` (`zmq::DONTWAIT`) under the mutex;
//!   `EAGAIN` returns the cached readout.
//! - **Drop / reset**: `reset` drops the socket under the mutex. `Drop` of
//!   the backend closes it through exclusive ownership (`zmq::Socket`'s
//!   `Drop` calls `zmq_close`).

use std::sync::{Mutex, MutexGuard};

use crate::zmq_readout::{max_readout_float_limit, parse_readout_packet};
use crate::{BackendError, IpcBackend};

/// Default ZeroMQ IPC endpoint for receiving compute data packets.
/// Can be overridden via environment variable `CORPUS_IPC_ZMQ_READOUT_IPC`.
const DEFAULT_READOUT_IPC: &str = "ipc:///tmp/corpus_ipc_readout.ipc";

/// Exclusive owner of a libzmq socket.
///
/// Inherits `Send` + `!Sync` from [`zmq::Socket`] (`zmq` 0.10 implements
/// `Send` and does not implement `Sync`). Socket operations on this wrapper
/// take `&mut self` so callers cannot invoke recv through a shared reference
/// without first obtaining exclusive access (the backend's [`Mutex`]).
struct ExclusiveSocket {
    socket: zmq::Socket,
}

impl ExclusiveSocket {
    fn recv_bytes_dontwait(&mut self) -> zmq::Result<Vec<u8>> {
        self.socket.recv_bytes(zmq::DONTWAIT)
    }
}

/// Deprecated compatibility name for [`ZmqIpcBackend`].
#[deprecated(since = "0.1.0", note = "renamed to ZmqIpcBackend")]
pub type ZmqRuntimeBackend = ZmqIpcBackend;

/// Generic IPC backend — subscribes to the remote compute's ZMQ PUB socket and
/// returns the latest compute readouts on each call.
///
/// Implements [`IpcBackend`]. This backend is a binary readout subscriber, not
/// a [`crate::HybridFlowBackend`]: it does not send or receive structured
/// `SpikeBatch` / `TraceBatch` messages.
///
/// # Concurrency
///
/// `ZmqIpcBackend` is `Send` + `Sync` so it can satisfy [`IpcBackend`]:
///
/// | Operation | Thread contract |
/// | --- | --- |
/// | Move to another thread | Supported. libzmq permits socket migration across a memory barrier; `Send` of this owned value is that barrier. |
/// | Share `&ZmqIpcBackend` | Allowed (`Sync`). Recv and `reset` close are serialized by an internal `Mutex`. Raw `zmq::Socket` is **not** `Sync` and is never shared. |
/// | `initialize` | `&mut self`. Creates, subscribes, and connects the SUB socket on the calling thread, then stores it under the mutex. Idempotent after success. |
/// | `process_batch` | `&mut self`. Non-blocking recv under the mutex; `EAGAIN` returns the cached readout. |
/// | `save_state` / `get_spike_states` | `&self`. Do not touch the socket. |
/// | `reset` | `&mut self`. Drops the SUB socket under the mutex. |
/// | `Drop` | Closes the socket through exclusive ownership of the backend (`zmq_close`). |
///
/// Concurrent `process_batch` still requires an outer lock (as
/// `corpus_ipc_server` does) because the trait method takes `&mut self`.
/// The inner mutex exists so this type can be `Sync` without an `unsafe impl
/// Sync` on the libzmq socket.
///
/// # Wire format
/// The packet consists of an 8-byte header followed by a variable number of
/// 4-byte floating-point values.
///
/// ```text
/// [0..8]   tick     i64 LE      monotonic tick counter
/// [8..]    readout  N×f32 LE    lobe outputs
/// ```
///
/// An **88-byte** frame is **tick + 20 floats** (not auto-split into 16 readouts
/// plus 4 neuromodulator scores). Historical producers that appended four
/// modulator scalars after sixteen readouts produce the same length as a valid
/// 20-float readout; this backend does not infer or strip that layout. Typed
/// neuromodulator extraction belongs to explicit higher-level parsing (see
/// [`crate::NeuromodulatorSnapshot`] JSON/`from_scores` ingress, not SUB auto-detection).
///
/// ## Bounds
///
/// [`parse_readout_packet`](crate::zmq_readout::parse_readout_packet) enforces a
/// maximum float count before resizing the decoded cache (default
/// [`DEFAULT_MAX_READOUT_FLOATS`](crate::zmq_readout::DEFAULT_MAX_READOUT_FLOATS),
/// override via [`ENV_MAX_READOUT_FLOATS`](crate::zmq_readout::ENV_MAX_READOUT_FLOATS)).
/// libzmq still allocates the raw received buffer; this cap applies only to the
/// decoded `Vec<f32>`.
pub struct ZmqIpcBackend {
    context: zmq::Context,
    /// Serialized so the backend is `Sync` without claiming `zmq::Socket: Sync`.
    sub_socket: Mutex<Option<ExclusiveSocket>>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub tick: i64,
}

impl ZmqIpcBackend {
    /// Create a new uninitialized ZMQ backend.
    ///
    /// Socket and subscription are established only on the first successful
    /// `initialize` call. Safe to construct even when the `zmq` feature is
    /// enabled but the external publisher is not yet running.
    pub fn new() -> Self {
        Self {
            context: zmq::Context::new(),
            sub_socket: Mutex::new(None),
            initialized: false,
            last_readout: Vec::new(),
            tick: 0,
        }
    }

    /// Monotonic tick counter of the last received packet.
    ///
    /// Updated on every successful `receive_readout` (inside `process_batch`).
    /// Useful for consumers that want to observe freshness without side effects.
    pub fn tick(&self) -> i64 {
        self.tick
    }

    fn lock_sub_socket(&self) -> MutexGuard<'_, Option<ExclusiveSocket>> {
        self.sub_socket
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Create, subscribe, connect, then store the SUB socket under the mutex.
    ///
    /// Socket construction and `connect` run on the calling thread *before*
    /// the value is stored. Only the store takes the lock.
    fn connect_sub(&mut self, endpoint: &str) -> Result<(), BackendError> {
        let socket = self
            .context
            .socket(zmq::SUB)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ SUB socket: {e}")))?;
        socket
            .set_subscribe(b"")
            .map_err(|e| BackendError::InitializationError(format!("ZMQ subscribe: {e}")))?;
        socket
            .set_rcvhwm(16)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ rcvhwm: {e}")))?;
        socket.connect(endpoint).map_err(|e| {
            BackendError::InitializationError(format!(
                "ZMQ connect to {endpoint}: {e} (is the IPC producer running?)"
            ))
        })?;
        *self.lock_sub_socket() = Some(ExclusiveSocket { socket });
        Ok(())
    }

    /// Test helper: initialize against a caller-chosen endpoint so live tests
    /// do not share `CORPUS_IPC_ZMQ_READOUT_IPC` / the default production path.
    #[cfg(test)]
    fn initialize_at(&mut self, endpoint: &str) -> Result<(), BackendError> {
        if self.initialized {
            return Ok(());
        }
        self.connect_sub(endpoint)?;
        self.initialized = true;
        Ok(())
    }

    /// Apply one received packet using the same validation path as production recv.
    fn ingest_readout_packet(&mut self, buf: &[u8]) -> Result<(), BackendError> {
        let max_floats = max_readout_float_limit();
        let (tick, readout) = parse_readout_packet(buf, max_floats)?;
        self.tick = tick;
        self.last_readout.clear();
        self.last_readout.extend_from_slice(&readout);
        Ok(())
    }

    fn receive_readout(&mut self) -> Result<Vec<f32>, BackendError> {
        let recv_result = {
            let mut guard = self.lock_sub_socket();
            let socket = guard.as_mut().ok_or_else(|| {
                BackendError::CommunicationError("SUB socket not connected".to_string())
            })?;
            socket.recv_bytes_dontwait()
        };

        match recv_result {
            Ok(buf) => self.ingest_readout_packet(&buf)?,
            Err(zmq::Error::EAGAIN) => {
                // No new packet available — return cached readout.
            }
            Err(e) => {
                return Err(BackendError::CommunicationError(format!(
                    "ZMQ recv failed: {e}"
                )));
            }
        }

        Ok(self.last_readout.clone())
    }

    /// Test/conformance hook: production ingest path without a live recv.
    #[doc(hidden)]
    pub fn apply_readout_packet_for_tests(&mut self, buf: &[u8]) -> Result<(), BackendError> {
        self.ingest_readout_packet(buf)
    }

    /// Test/conformance hook: observe `(tick, readout cache)` without recv.
    #[doc(hidden)]
    pub fn readout_cache_snapshot_for_tests(&self) -> (i64, Vec<f32>) {
        (self.tick, self.last_readout.clone())
    }
}

impl Default for ZmqIpcBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IpcBackend for ZmqIpcBackend {
    /// Process a dynamic slice of input signals through the compute backend.
    ///
    /// For ZMQ this ignores the `inputs` (the backend is a readout subscriber)
    /// and returns the latest packet or cached value.
    ///
    /// # Errors
    /// Returns `InitializationError` if the SUB socket is not connected.
    fn process_batch(&mut self, _inputs: &[f32]) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "ZmqIpcBackend not initialized — call initialize() first".to_string(),
            ));
        }
        self.receive_readout()
    }

    /// Initialise the ZMQ SUB socket and connect to the readout endpoint.
    ///
    /// Endpoint may be overridden by `CORPUS_IPC_ZMQ_READOUT_IPC`.
    ///
    /// Idempotent: second call is a no-op once connected.
    ///
    /// To switch to a different endpoint at runtime (e.g. after changing the
    /// `CORPUS_IPC_ZMQ_READOUT_IPC` env var), call `reset()` first to clear
    /// the initialized flag and drop the current socket, then call `initialize()`
    /// again. Without `reset()`, a second `initialize()` is a no-op.
    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        if self.initialized {
            return Ok(());
        }
        let endpoint = std::env::var("CORPUS_IPC_ZMQ_READOUT_IPC")
            .unwrap_or_else(|_| DEFAULT_READOUT_IPC.to_string());
        self.connect_sub(&endpoint)?;
        self.initialized = true;
        println!("[zmq-ipc] Connected to IPC producer at {}", endpoint);
        Ok(())
    }

    /// Persist current model state (delegated to remote if supported).
    /// Current ZMQ implementation is read-only; this is a no-op.
    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        // State lives in the external compute process; this is a no-op.
        Ok(())
    }

    /// Derive spike states from the last readout vector.
    ///
    /// Values > 0.5 are treated as spiked (true). This is an approximation
    /// since the ZMQ readout is a scalar activation vector, not explicit spikes.
    /// (RustBackend returns an always-empty Vec because it is stateless.)
    fn get_spike_states(&self) -> Vec<bool> {
        self.last_readout.iter().map(|&v| v > 0.5).collect()
    }

    /// Reset cached readout state. Does not affect the remote process.
    ///
    /// As a side effect, clears the `initialized` flag and drops the current
    /// SUB socket (if any). This allows a subsequent call to `initialize()`
    /// to re-establish the connection (e.g. after changing
    /// `CORPUS_IPC_ZMQ_READOUT_IPC` at runtime).
    fn reset(&mut self) -> Result<(), BackendError> {
        self.last_readout.clear();
        self.tick = 0;
        self.initialized = false;
        *self.lock_sub_socket() = None;
        println!("[zmq-ipc] Readout cache reset; will re-initialize on next call");
        Ok(())
    }
}

// Compile-time record of the intended Send/Sync surface. These functions are
// never called at runtime; rustc still type-checks their bodies whenever the
// `zmq` feature is enabled.
const _: fn() = _assert_intended_send_sync_surface;

fn _assert_intended_send_sync_surface() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_ipc_backend<T: IpcBackend>() {}

    assert_send::<zmq::Socket>();
    assert_send::<ExclusiveSocket>();
    assert_send::<ZmqIpcBackend>();
    assert_sync::<ZmqIpcBackend>();
    assert_ipc_backend::<ZmqIpcBackend>();

    // Negative assertions must name a concrete type. A generic helper like
    // `fn check<T>()` would infer `_` as `NotSyncToken` from the blanket impl
    // and never become ambiguous, even for `T: Sync`.
    let _ = <zmq::Socket as _assert_not_sync::IfSyncThenAmbiguous<_>>::witness;
    let _ = <ExclusiveSocket as _assert_not_sync::IfSyncThenAmbiguous<_>>::witness;
}

/// Overlapping-impl witness that a concrete type is not `Sync`.
///
/// If the type is `Sync`, both impls apply and turbofish `_` cannot be
/// inferred. If it is `!Sync`, only the blanket impl applies.
mod _assert_not_sync {
    pub trait IfSyncThenAmbiguous<Marker> {
        fn witness() {}
    }

    pub struct NotSyncToken;
    pub struct SyncToken;

    impl<T: ?Sized> IfSyncThenAmbiguous<NotSyncToken> for T {}
    impl<T: ?Sized + Sync> IfSyncThenAmbiguous<SyncToken> for T {}
}

// ── Packet-parsing unit tests (no live ZMQ socket needed) ────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn unique_test_endpoint() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "ipc:///tmp/corpus_ipc_zmq_test_{}_{}.ipc",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn make_packet(tick: i64, readout: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + readout.len() * 4);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in readout {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parse_dynamic_packet_via_production_ingest() {
        let readout: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let tick: i64 = 42_000;
        let buf = make_packet(tick, &readout);

        let mut b = ZmqIpcBackend::new();
        b.apply_readout_packet_for_tests(&buf)
            .expect("valid 20-float frame");

        assert_eq!(b.tick, tick);
        assert_eq!(b.last_readout.len(), 20);
        for (i, val) in readout.iter().enumerate().take(20) {
            assert!((b.last_readout[i] - val).abs() < 1e-5);
        }
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_alias_is_same_type() {
        fn same(backend: crate::ZmqRuntimeBackend) -> ZmqIpcBackend {
            backend
        }
        let _ = same(ZmqIpcBackend::new());
    }

    #[test]
    fn malformed_packet_does_not_mutate_state() {
        let mut b = ZmqIpcBackend::new();
        b.last_readout = vec![1.0, 2.0];
        b.tick = 100;

        let initial_readout = b.last_readout.clone();
        let initial_tick = b.tick;

        let bad_buf: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00];
        let err = b
            .apply_readout_packet_for_tests(bad_buf)
            .expect_err("truncated packet must fail");
        assert!(
            matches!(err, BackendError::CommunicationError(_)),
            "malformed framing is not EAGAIN: {err:?}"
        );

        assert_eq!(b.last_readout, initial_readout);
        assert_eq!(b.tick, initial_tick);
    }

    #[test]
    fn over_limit_packet_is_invalid_input_not_communication() {
        use crate::zmq_readout::parse_readout_packet;

        let max = 4;
        let buf = make_packet(1, &[0.0; 5]);
        let err = parse_readout_packet(&buf, max).expect_err("five floats exceeds cap of four");
        assert!(matches!(err, BackendError::InvalidInput(_)));
    }

    #[test]
    fn initialize_nonblocking_recv_and_drop() {
        let mut backend = ZmqIpcBackend::new();
        backend
            .initialize_at(&unique_test_endpoint())
            .expect("ZMQ connect is asynchronous; a publisher need not be bound");
        // DONTWAIT with no publisher returns EAGAIN and the empty cache.
        let out = backend.process_batch(&[1.0]).unwrap();
        assert!(out.is_empty());
        assert_eq!(backend.tick(), 0);
        drop(backend);
    }

    #[test]
    fn connected_backend_can_move_to_owner_thread() {
        let mut backend = ZmqIpcBackend::new();
        backend.initialize_at(&unique_test_endpoint()).unwrap();
        let readout = std::thread::spawn(move || {
            let out = backend.process_batch(&[]).unwrap();
            drop(backend);
            out
        })
        .join()
        .expect("owner thread panicked");
        assert!(readout.is_empty());
    }

    #[test]
    fn uninitialized_backend_can_move_then_initialize() {
        let backend = ZmqIpcBackend::new();
        let endpoint = unique_test_endpoint();
        std::thread::spawn(move || {
            let mut backend = backend;
            backend.initialize_at(&endpoint).unwrap();
            let out = backend.process_batch(&[]).unwrap();
            assert!(out.is_empty());
        })
        .join()
        .expect("owner thread panicked");
    }

    #[test]
    fn repeated_construct_initialize_process_drop() {
        for _ in 0..8 {
            let mut backend = ZmqIpcBackend::new();
            backend.initialize_at(&unique_test_endpoint()).unwrap();
            let out = backend.process_batch(&[]).unwrap();
            assert!(out.is_empty());
            backend.reset().unwrap();
            drop(backend);
        }
    }

    #[test]
    fn backend_is_usable_behind_arc_mutex() {
        let backend = Arc::new(Mutex::new(ZmqIpcBackend::new()));
        {
            let mut guard = backend.lock().unwrap();
            guard.initialize_at(&unique_test_endpoint()).unwrap();
            assert!(guard.process_batch(&[]).unwrap().is_empty());
        }
        let moved = Arc::clone(&backend);
        std::thread::spawn(move || {
            let mut guard = moved.lock().unwrap();
            assert!(guard.process_batch(&[]).unwrap().is_empty());
            guard.reset().unwrap();
        })
        .join()
        .expect("mutex owner thread panicked");
    }

    #[test]
    fn thread_safety_surface_is_the_documented_one() {
        _assert_intended_send_sync_surface();
    }
}
