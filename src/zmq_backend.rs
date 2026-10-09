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
//! - **Process**: drains non-blocking `recv` calls (`zmq::DONTWAIT`) under the
//!   mutex until `EAGAIN` or the 64-message / 5 ms budget, then returns the
//!   newest valid readout received.
//! - **Drop / reset**: `reset` drops the socket under the mutex. `Drop` of
//!   the backend closes it through exclusive ownership (`zmq::Socket`'s
//!   `Drop` calls `zmq_close`).

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

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

const READOUT_RCVHWM: i32 = 16;
/// Cap on non-blocking recv iterations per `process_batch` so a saturated
/// publisher cannot pin the backend mutex forever waiting for EAGAIN.
const RECEIVE_DRAIN_MAX_MESSAGES: u32 = 64; // >= READOUT_RCVHWM*4; keep const-friendly
/// Soft wall-clock budget for the same drain (parse time included).
const RECEIVE_DRAIN_MAX_DURATION: Duration = Duration::from_millis(5);

/// Default initial reconnect interval (`ZMQ_RECONNECT_IVL`), in milliseconds.
/// Matches the libzmq default of 100 ms and is applied explicitly so the
/// value is documented and overrideable via [`ENV_RECONNECT_IVL_MS`].
pub const DEFAULT_RECONNECT_IVL_MS: i32 = 100;
/// Default maximum reconnect interval (`ZMQ_RECONNECT_IVL_MAX`), in
/// milliseconds. libzmq's default of `0` disables exponential backoff;
/// this crate instead caps backoff at a finite 5 s so a long-down publisher
/// does not pin reconnect attempts at the initial interval.
pub const DEFAULT_RECONNECT_IVL_MAX_MS: i32 = 5_000;
/// Environment variable overriding [`DEFAULT_RECONNECT_IVL_MS`]
/// (`ZMQ_RECONNECT_IVL`). Positive milliseconds only; a missing,
/// unparseable, or non-positive value falls back to the default.
pub const ENV_RECONNECT_IVL_MS: &str = "CORPUS_IPC_ZMQ_RECONNECT_IVL_MS";
/// Environment variable overriding [`DEFAULT_RECONNECT_IVL_MAX_MS`]
/// (`ZMQ_RECONNECT_IVL_MAX`). Positive milliseconds only; a missing,
/// unparseable, or non-positive value falls back to the default.
pub const ENV_RECONNECT_IVL_MAX_MS: &str = "CORPUS_IPC_ZMQ_RECONNECT_IVL_MAX_MS";

/// Resolved `ZMQ_RECONNECT_IVL`/`ZMQ_RECONNECT_IVL_MAX` (milliseconds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReconnectConfig {
    ivl_ms: i32,
    ivl_max_ms: i32,
}

fn env_i32_or(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn reconnect_config_from_env() -> ReconnectConfig {
    ReconnectConfig {
        ivl_ms: env_i32_or(ENV_RECONNECT_IVL_MS, DEFAULT_RECONNECT_IVL_MS),
        ivl_max_ms: env_i32_or(ENV_RECONNECT_IVL_MAX_MS, DEFAULT_RECONNECT_IVL_MAX_MS),
    }
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
/// returns the latest compute readout available on each call. The SUB socket
/// has a receive high-water mark of 16 packets. `process_batch` drains the
/// socket until `EAGAIN` or the 64-message / 5 ms budget and applies only the
/// last valid packet; use
/// [`ZmqIpcBackend::skipped_readouts`] to observe valid packets superseded by
/// that latest-value policy. Drops performed internally by libzmq at the HWM
/// are silent and therefore cannot be included in the counter.
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
/// | `process_batch` | `&mut self`. Drains non-blocking recv calls under the mutex; `EAGAIN` returns the newest valid readout or the cache. |
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
/// [`parse_readout_packet`] enforces a
/// maximum float count before resizing the decoded cache (default
/// [`DEFAULT_MAX_READOUT_FLOATS`](crate::zmq_readout::DEFAULT_MAX_READOUT_FLOATS),
/// override via [`ENV_MAX_READOUT_FLOATS`](crate::zmq_readout::ENV_MAX_READOUT_FLOATS)).
/// libzmq still allocates the raw received buffer; this cap applies only to the
/// decoded `Vec<f32>`.
///
/// ## Reconnect and close behavior
///
/// The SUB socket is configured with `ZMQ_RECONNECT_IVL` /
/// `ZMQ_RECONNECT_IVL_MAX` (defaults [`DEFAULT_RECONNECT_IVL_MS`] and
/// [`DEFAULT_RECONNECT_IVL_MAX_MS`], overridable via [`ENV_RECONNECT_IVL_MS`]
/// and [`ENV_RECONNECT_IVL_MAX_MS`]) *before* `connect`. Recovery after the
/// publisher starts late, stops, or restarts at the same endpoint is handled
/// entirely by libzmq's reconnect state machine — including re-sending the
/// stored `subscribe` filter on each new connection — so there is no
/// crate-owned retry thread and `process_batch` stays non-blocking.
///
/// `reset` and `Drop` close the socket via `zmq_close` with libzmq's default
/// `ZMQ_LINGER`. A SUB socket has no outbound queue, so no unsent data can
/// wait on linger; close takes effect once in-flight wire reads drain.
pub struct ZmqIpcBackend {
    context: zmq::Context,
    /// Serialized so the backend is `Sync` without claiming `zmq::Socket: Sync`.
    sub_socket: Mutex<Option<ExclusiveSocket>>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub tick: i64,
    skipped_readouts: u64,
    malformed_readouts: u64,
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
            skipped_readouts: 0,
            malformed_readouts: 0,
        }
    }

    /// Monotonic tick counter of the last received packet.
    ///
    /// Updated on every successful `receive_readout` (inside `process_batch`).
    /// Useful for consumers that want to observe freshness without side effects.
    pub fn tick(&self) -> i64 {
        self.tick
    }

    /// Number of valid queued readouts discarded in favor of a newer readout.
    ///
    /// This is a lower bound on transport loss: libzmq silently drops packets
    /// when the receive HWM is exceeded, before this backend can count them.
    pub fn skipped_readouts(&self) -> u64 {
        self.skipped_readouts
    }

    /// Number of malformed readout frames rejected without changing the cache.
    pub fn malformed_readouts(&self) -> u64 {
        self.malformed_readouts
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
    fn connect_sub_with(
        &mut self,
        endpoint: &str,
        cfg: ReconnectConfig,
    ) -> Result<(), BackendError> {
        let socket = self
            .context
            .socket(zmq::SUB)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ SUB socket: {e}")))?;
        socket
            .set_subscribe(b"")
            .map_err(|e| BackendError::InitializationError(format!("ZMQ subscribe: {e}")))?;
        socket
            .set_rcvhwm(READOUT_RCVHWM)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ rcvhwm: {e}")))?;
        // Reconnect options must be applied before `connect`: libzmq reads
        // them when establishing each connection attempt, so setting them
        // afterwards would leave the first reconnect cycle at the old values.
        socket
            .set_reconnect_ivl(cfg.ivl_ms)
            .map_err(|e| BackendError::InitializationError(format!("ZMQ reconnect_ivl: {e}")))?;
        socket.set_reconnect_ivl_max(cfg.ivl_max_ms).map_err(|e| {
            BackendError::InitializationError(format!("ZMQ reconnect_ivl_max: {e}"))
        })?;
        socket.connect(endpoint).map_err(|e| {
            BackendError::InitializationError(format!(
                "ZMQ connect to {endpoint}: {e} (is the IPC producer running?)"
            ))
        })?;
        *self.lock_sub_socket() = Some(ExclusiveSocket { socket });
        Ok(())
    }

    fn connect_sub(&mut self, endpoint: &str) -> Result<(), BackendError> {
        self.connect_sub_with(endpoint, reconnect_config_from_env())
    }

    /// Test helper: initialize against a caller-chosen endpoint so live tests
    /// do not share `CORPUS_IPC_ZMQ_READOUT_IPC` / the default production path.
    /// Uses short reconnect intervals (env vars are process-global and unsafe
    /// to mutate from parallel tests) so restart tests stay fast.
    #[cfg(test)]
    fn initialize_at(&mut self, endpoint: &str) -> Result<(), BackendError> {
        if self.initialized {
            return Ok(());
        }
        self.connect_sub_with(
            endpoint,
            ReconnectConfig {
                ivl_ms: 10,
                ivl_max_ms: 100,
            },
        )?;
        self.initialized = true;
        Ok(())
    }

    /// Apply one received packet using an explicit float cap.
    fn ingest_readout_packet_with_limit(
        &mut self,
        buf: &[u8],
        max_floats: usize,
    ) -> Result<(), BackendError> {
        let (tick, readout) = parse_readout_packet(buf, max_floats)?;
        self.tick = tick;
        self.last_readout.clear();
        self.last_readout.extend_from_slice(&readout);
        Ok(())
    }

    /// Apply one received packet using the configured environment float cap.
    fn ingest_readout_packet(&mut self, buf: &[u8]) -> Result<(), BackendError> {
        self.ingest_readout_packet_with_limit(buf, max_readout_float_limit())
    }

    fn drain_budget_exhausted(received: u32, started: Instant) -> bool {
        if received >= RECEIVE_DRAIN_MAX_MESSAGES {
            return true;
        }
        if received == 0 {
            return false;
        }
        started.elapsed() >= RECEIVE_DRAIN_MAX_DURATION
    }

    fn try_recv_readout_packet(&self) -> Result<Option<Vec<u8>>, BackendError> {
        let mut guard = self.lock_sub_socket();
        let socket = guard.as_mut().ok_or_else(|| {
            BackendError::CommunicationError("SUB socket not connected".to_string())
        })?;
        match socket.recv_bytes_dontwait() {
            Ok(buf) => Ok(Some(buf)),
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(BackendError::CommunicationError(format!(
                "ZMQ recv failed: {e}"
            ))),
        }
    }

    fn parse_received_readout(&mut self, buf: &[u8]) -> Result<(i64, Vec<f32>), BackendError> {
        parse_readout_packet(buf, max_readout_float_limit()).inspect_err(|_| {
            self.malformed_readouts = self.malformed_readouts.saturating_add(1);
        })
    }

    fn receive_readout(&mut self) -> Result<Vec<f32>, BackendError> {
        let mut newest = None;
        let started = Instant::now();
        let mut received = 0u32;
        loop {
            if Self::drain_budget_exhausted(received, started) {
                // Bound reached with the socket possibly still non-empty: keep
                // the newest valid frame seen this call; leftover frames remain
                // for a later process_batch (or are dropped by RCVHWM).
                break;
            }
            let Some(buf) = self.try_recv_readout_packet()? else {
                break;
            };
            received = received.saturating_add(1);
            let parsed = self.parse_received_readout(&buf)?;
            if newest.replace(parsed).is_some() {
                self.skipped_readouts = self.skipped_readouts.saturating_add(1);
            }
        }

        if let Some((tick, readout)) = newest {
            self.tick = tick;
            self.last_readout = readout;
        }

        Ok(self.last_readout.clone())
    }

    /// Test/conformance hook: production ingest path without a live recv.
    #[doc(hidden)]
    pub fn apply_readout_packet_for_tests(&mut self, buf: &[u8]) -> Result<(), BackendError> {
        self.ingest_readout_packet(buf)
    }

    /// Test/conformance hook: production ingest path with explicit float limit.
    #[doc(hidden)]
    pub fn apply_readout_packet_with_limit_for_tests(
        &mut self,
        buf: &[u8],
        max_floats: usize,
    ) -> Result<(), BackendError> {
        self.ingest_readout_packet_with_limit(buf, max_floats)
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
    /// Returns `InitializationError` if the SUB socket is not connected,
    /// `CommunicationError` for malformed framing or receive failure, and
    /// `InvalidInput` for a readout above the configured float limit. A
    /// rejected frame leaves the cached readout unchanged.
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
    /// Endpoint may be overridden by `CORPUS_IPC_ZMQ_READOUT_IPC`; reconnect
    /// intervals by [`ENV_RECONNECT_IVL_MS`] / [`ENV_RECONNECT_IVL_MAX_MS`].
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
    /// SUB socket (if any) under the socket mutex; the drop runs `zmq_close`
    /// with libzmq's default `ZMQ_LINGER` (no outbound queue exists on SUB,
    /// so nothing blocks on unsent data). This allows a subsequent call to
    /// `initialize()` to re-establish the connection (e.g. after changing
    /// `CORPUS_IPC_ZMQ_READOUT_IPC` at runtime). This is the explicit path
    /// for endpoint changes or forced manual recovery — automatic recovery
    /// from publisher restarts is libzmq's reconnect behavior instead.
    fn reset(&mut self) -> Result<(), BackendError> {
        self.last_readout.clear();
        self.tick = 0;
        self.skipped_readouts = 0;
        self.malformed_readouts = 0;
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

    /// Wait until `pred` holds or `timeout` elapses (ZMQ delivery is async).
    fn wait_until(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if pred() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        pred()
    }

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

    fn connected_pub_sub() -> (zmq::Socket, ZmqIpcBackend) {
        let endpoint = unique_test_endpoint();
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind(&endpoint).unwrap();

        let mut backend = ZmqIpcBackend::new();
        backend.initialize_at(&endpoint).unwrap();

        // PUB/SUB subscriptions propagate asynchronously. A warm-up loop
        // makes the burst assertions independent of the slow-joiner window.
        for _ in 0..100 {
            publisher.send(make_packet(1, &[1.0]), 0).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
            backend.process_batch(&[]).unwrap();
            if backend.tick() == 1 {
                return (publisher, backend);
            }
        }
        panic!("ZMQ subscription did not become ready");
    }

    /// Send `packet` (with `tick`) until the backend observes that tick or the
    /// deadline passes. Used by reconnect tests: frames sent before a (re)bound
    /// publisher sees the subscription are dropped, so sends must repeat.
    fn publish_until_seen(
        publisher: &zmq::Socket,
        backend: &mut ZmqIpcBackend,
        tick: i64,
        timeout: Duration,
    ) -> bool {
        let packet = make_packet(tick, &[tick as f32]);
        wait_until(timeout, || {
            publisher.send(packet.clone(), 0).unwrap();
            backend.process_batch(&[]).unwrap();
            backend.tick() == tick
        })
    }

    /// A backend connected before any publisher exists recovers purely via
    /// libzmq reconnect (`ZMQ_RECONNECT_IVL`) once the publisher binds — no
    /// crate-owned retry thread and no `reset()`/`initialize()` cycle.
    #[test]
    fn late_publisher_startup_recovers_via_libzmq_reconnect() {
        let endpoint = unique_test_endpoint();
        let mut backend = ZmqIpcBackend::new();
        backend
            .initialize_at(&endpoint)
            .expect("connect is asynchronous before the publisher binds");
        assert!(
            backend.process_batch(&[]).unwrap().is_empty(),
            "no publisher yet: empty cache on EAGAIN"
        );

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind(&endpoint).unwrap();

        assert!(
            publish_until_seen(&publisher, &mut backend, 7, Duration::from_secs(5)),
            "timed out waiting for libzmq reconnect after late publisher bind"
        );
        assert_eq!(backend.last_readout, vec![7.0]);
    }

    /// The publisher is dropped and a fresh PUB binds the same IPC endpoint;
    /// the SUB socket reconnects and re-subscribes without a `reset()`.
    #[test]
    fn publisher_restart_on_same_ipc_endpoint_recovers() {
        publisher_restart_on_same_endpoint_recovers(&unique_test_endpoint());
    }

    /// Same restart scenario over a TCP loopback endpoint.
    #[test]
    fn publisher_restart_on_same_tcp_endpoint_recovers() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        publisher_restart_on_same_endpoint_recovers(&format!("tcp://127.0.0.1:{port}"));
    }

    fn publisher_restart_on_same_endpoint_recovers(endpoint: &str) {
        let context = zmq::Context::new();
        let mut backend = ZmqIpcBackend::new();
        backend.initialize_at(endpoint).unwrap();

        {
            let first = context.socket(zmq::PUB).unwrap();
            first.bind(endpoint).unwrap();
            assert!(
                publish_until_seen(&first, &mut backend, 1, Duration::from_secs(5)),
                "timed out waiting for the first publisher"
            );
            // `Drop` runs `zmq_close` with default linger; the SUB socket's
            // peer is now gone and libzmq starts the reconnect backoff.
        }

        let second = context.socket(zmq::PUB).unwrap();
        // `zmq_close` tears the old binding down on a libzmq I/O thread, so a
        // TCP endpoint can briefly report EADDRINUSE; retry the rebind.
        let bound = wait_until(Duration::from_secs(5), || second.bind(endpoint).is_ok());
        assert!(bound, "timed out rebinding {endpoint} after publisher drop");
        assert!(
            publish_until_seen(&second, &mut backend, 2, Duration::from_secs(10)),
            "timed out waiting for libzmq reconnect after publisher restart"
        );
        assert_eq!(backend.last_readout, vec![2.0]);
    }

    /// `reset()` closes the old socket and clears state; `initialize` may then
    /// connect to a replacement endpoint — the explicit manual-recovery path.
    #[test]
    fn reset_then_initialize_at_replacement_endpoint() {
        let stale_endpoint = unique_test_endpoint();
        let live_endpoint = unique_test_endpoint();
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind(&live_endpoint).unwrap();

        let mut backend = ZmqIpcBackend::new();
        backend.initialize_at(&stale_endpoint).unwrap();
        backend.reset().unwrap();
        assert_eq!(backend.readout_cache_snapshot_for_tests(), (0, vec![]));
        backend.initialize_at(&live_endpoint).unwrap();

        assert!(
            publish_until_seen(&publisher, &mut backend, 5, Duration::from_secs(5)),
            "timed out waiting for delivery on the replacement endpoint"
        );
        assert_eq!(backend.last_readout, vec![5.0]);
    }

    #[test]
    fn parse_dynamic_packet_via_production_ingest() {
        let max_cap = crate::zmq_readout::max_readout_float_limit();
        let count = max_cap.clamp(1, 20);
        let readout: Vec<f32> = (0..count).map(|i| i as f32 * 0.1).collect();
        let tick: i64 = 42_000;
        let buf = make_packet(tick, &readout);

        let mut b = ZmqIpcBackend::new();
        b.apply_readout_packet_for_tests(&buf)
            .expect("valid dynamic frame within configured limit");

        assert_eq!(b.tick, tick);
        assert_eq!(b.last_readout.len(), count);
        for (i, val) in readout.iter().enumerate().take(count) {
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
    fn process_batch_drains_beyond_hwm_to_newest_accepted_frame() {
        let (publisher, mut backend) = connected_pub_sub();
        let packet_count = READOUT_RCVHWM as i64 * 4;
        let skipped_before = backend.skipped_readouts();

        for tick in 2..=packet_count + 1 {
            publisher
                .send(make_packet(tick, &[tick as f32]), 0)
                .unwrap();
        }
        assert!(
            wait_until(Duration::from_millis(500), || {
                backend.process_batch(&[]).unwrap();
                backend.tick() > 1 && backend.skipped_readouts() > skipped_before
            }),
            "timed out waiting for a burst with superseded readouts"
        );
        let output = backend.last_readout.clone();
        let newest_accepted_tick = backend.tick();
        assert!(newest_accepted_tick > 1);
        assert_eq!(output, vec![newest_accepted_tick as f32]);
        assert!(backend.skipped_readouts() > 0);

        // Delivery is asynchronous, so later calls may still see frames that
        // had not reached the SUB socket during the first bounded drain.
    }

    #[test]
    fn malformed_frame_in_drained_burst_fails_closed_without_applying_cache() {
        let (publisher, mut backend) = connected_pub_sub();
        let malformed_before = backend.malformed_readouts();
        publisher.send(make_packet(2, &[2.0]), 0).unwrap();
        publisher
            .send([0xde, 0xad, 0xbe, 0xef].as_slice(), 0)
            .unwrap();
        publisher.send(make_packet(3, &[3.0]), 0).unwrap();
        assert!(
            wait_until(Duration::from_millis(500), || {
                let before = backend.readout_cache_snapshot_for_tests();
                match backend.process_batch(&[]) {
                    Err(BackendError::CommunicationError(_)) => {
                        assert_eq!(backend.readout_cache_snapshot_for_tests(), before);
                        true
                    }
                    Ok(_) => false,
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            }),
            "timed out waiting for malformed frame"
        );
        assert!(
            wait_until(Duration::from_millis(500), || {
                backend.process_batch(&[]).unwrap() == vec![3.0]
            }),
            "timed out waiting for tick-3 frame"
        );
        assert_eq!(backend.tick(), 3);
        assert_eq!(backend.malformed_readouts() - malformed_before, 1);
        backend.reset().unwrap();
        assert_eq!(backend.malformed_readouts(), 0);
        assert_eq!(backend.skipped_readouts(), 0);
        assert_eq!(backend.tick(), 0);
    }

    #[test]
    fn over_limit_frame_in_drained_burst_fails_closed() {
        let (publisher, mut backend) = connected_pub_sub();
        let before = backend.readout_cache_snapshot_for_tests();
        let max_floats = max_readout_float_limit();
        publisher
            .send(make_packet(2, &vec![0.0; max_floats + 1]), 0)
            .unwrap();
        assert!(
            wait_until(Duration::from_millis(500), || {
                match backend.process_batch(&[]) {
                    Err(BackendError::InvalidInput(_)) => true,
                    Ok(_) => false,
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            }),
            "timed out waiting for over-limit frame"
        );
        assert_eq!(backend.readout_cache_snapshot_for_tests(), before);
        assert_eq!(backend.malformed_readouts(), 1);
    }

    #[test]
    fn process_batch_before_initialize_returns_initialization_error() {
        let mut backend = ZmqIpcBackend::new();
        let err = backend.process_batch(&[]).unwrap_err();
        assert!(matches!(err, BackendError::InitializationError(_)));
    }

    #[test]
    fn initialize_without_publisher_is_nonblocking() {
        let mut backend = ZmqIpcBackend::new();
        backend.initialize(None).unwrap();
        assert!(backend.process_batch(&[]).unwrap().is_empty());
        backend.reset().unwrap();
        assert!(matches!(
            backend.process_batch(&[]),
            Err(BackendError::InitializationError(_))
        ));
    }

    #[test]
    fn missing_sub_socket_reports_communication_error() {
        let backend = ZmqIpcBackend::new();
        assert!(matches!(
            backend.try_recv_readout_packet(),
            Err(BackendError::CommunicationError(_))
        ));
    }

    #[test]
    fn get_spike_states_thresholds_last_readout() {
        let (publisher, mut backend) = connected_pub_sub();
        let max_cap = crate::zmq_readout::max_readout_float_limit();
        let values: Vec<f32> = if max_cap >= 3 {
            vec![0.25, 0.75, 0.51]
        } else {
            vec![0.75]
        };
        let expected: Vec<bool> = values.iter().map(|&v| v > 0.5).collect();
        let expected_len = values.len();

        publisher.send(make_packet(9, &values), 0).unwrap();
        assert!(
            wait_until(Duration::from_millis(200), || {
                backend
                    .process_batch(&[])
                    .is_ok_and(|out| out.len() == expected_len)
            }),
            "timed out waiting for multi-channel readout"
        );
        assert_eq!(backend.get_spike_states(), expected);
    }

    #[test]
    fn receive_drain_bound_returns_under_sustained_publish_load() {
        let (publisher, mut backend) = connected_pub_sub();
        let flooder = std::thread::spawn(move || {
            for tick in 2..=512i64 {
                let _ = publisher.send(make_packet(tick, &[tick as f32]), zmq::DONTWAIT);
            }
        });
        std::thread::sleep(Duration::from_millis(10));
        let started = Instant::now();
        let output = backend.process_batch(&[]).unwrap();
        let elapsed = started.elapsed();
        flooder.join().expect("flooder panicked");
        assert!(
            elapsed < Duration::from_millis(100),
            "bounded drain must return promptly under sustained publish load"
        );
        assert!(backend.tick() > 1);
        assert_eq!(output[0], backend.tick() as f32);
    }

    #[test]
    fn drain_budget_always_allows_first_receive_attempt() {
        let expired = Instant::now() - RECEIVE_DRAIN_MAX_DURATION;
        assert!(!ZmqIpcBackend::drain_budget_exhausted(0, expired));
        assert!(ZmqIpcBackend::drain_budget_exhausted(1, expired));
        assert!(ZmqIpcBackend::drain_budget_exhausted(
            RECEIVE_DRAIN_MAX_MESSAGES,
            Instant::now()
        ));
    }

    #[test]
    fn over_limit_packet_is_invalid_input_without_mutating_state() {
        let max = 4;
        let buf = make_packet(1, &[0.0; 5]);
        let mut b = ZmqIpcBackend::new();
        b.last_readout = vec![1.0, 2.0];
        b.tick = 100;
        let before = b.readout_cache_snapshot_for_tests();

        let err = b
            .apply_readout_packet_with_limit_for_tests(&buf, max)
            .expect_err("five floats exceeds cap of four");
        assert!(matches!(err, BackendError::InvalidInput(_)));
        assert_eq!(b.readout_cache_snapshot_for_tests(), before);
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
