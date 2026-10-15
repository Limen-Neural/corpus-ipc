// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Mutex, MutexGuard};

use crate::validation::Validate;
use crate::{
    CanonicalEncodeError, IpcMessage, decode_ipc_message_json, encode_canonical_ipc_message,
};

/// Why a ZeroMQ message was not a single structured frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FramingFault {
    /// The frame contained no bytes.
    #[error("message was empty")]
    Empty,
    /// libzmq delivered more than one frame for the message.
    #[error("message had multiple ZeroMQ frames; exactly one JSON envelope frame is required")]
    Multipart,
}

/// Failures from the structured ZeroMQ publisher and subscriber.
///
/// Outcomes are matchable. Library paths return one of these variants instead
/// of panicking on malformed input, version mismatch, high-water-mark pressure,
/// or a closed socket.
#[derive(Debug, thiserror::Error)]
pub enum ZmqMessageError {
    /// [`ZmqTransportConfig`] rejected the caller's settings before any socket call.
    #[error("invalid transport config: {0}")]
    InvalidConfig(&'static str),

    /// Socket creation, option setup, bind, or connect failed.
    #[error("ZeroMQ socket setup failed: {0}")]
    Setup(String),

    /// The message failed [`Validate`] before encoding.
    #[error("payload failed validation: {0}")]
    Validation(#[from] crate::ValidationError),

    /// Canonical encoding failed after validation passed.
    #[error("canonical encoding failed: {0}")]
    Encode(CanonicalEncodeError),

    /// The frame was not an acceptable [`crate::WireEnvelope`] or legacy payload.
    #[error("envelope decode failed: {0}")]
    Envelope(#[from] crate::EnvelopeError),

    /// The ZeroMQ message was not exactly one non-empty frame.
    #[error("unexpected ZeroMQ framing: {0}")]
    UnexpectedFraming(FramingFault),

    /// Non-blocking receive found no message (`EAGAIN`).
    #[error("no message available (EAGAIN)")]
    WouldBlock,

    /// Non-blocking send returned `EAGAIN` because the outbound queue is full.
    ///
    /// PUB sockets more often drop silently when the high-water mark is reached
    /// and `send` still returns `Ok`. This variant is the case where libzmq
    /// reports the pressure instead of dropping.
    #[error("outbound high-water mark is full (EAGAIN on send)")]
    HighWaterMark,

    /// `close` already dropped the socket, or it was never stored.
    #[error("ZeroMQ socket is closed")]
    Closed,

    /// A libzmq send or receive error other than `EAGAIN`.
    #[error("ZeroMQ transport error: {0}")]
    Transport(String),
}

/// Endpoint and socket options for one structured PUB or SUB socket.
///
/// Options are applied before `bind` or `connect`. High-water marks of `0`
/// are legal and mean "do not queue". Linger `-1` means wait forever on close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqTransportConfig {
    /// ZeroMQ endpoint (`ipc://…` or `tcp://…`). The publisher binds it; the
    /// subscriber connects to it.
    pub endpoint: String,
    /// `ZMQ_SNDHWM`. Must be `>= 0`.
    pub send_hwm: i32,
    /// `ZMQ_RCVHWM`. Must be `>= 0`.
    pub receive_hwm: i32,
    /// `ZMQ_LINGER` in milliseconds. `-1` waits forever; `0` discards unsent
    /// frames on close. Must be `>= -1`.
    pub linger_ms: i32,
    /// `ZMQ_RECONNECT_IVL` in milliseconds. Must be `> 0`.
    pub reconnect_ivl_ms: i32,
    /// `ZMQ_RECONNECT_IVL_MAX` in milliseconds. Must be `> 0` and at least
    /// [`Self::reconnect_ivl_ms`]. libzmq ignores a cap below the interval.
    pub reconnect_ivl_max_ms: i32,
}

impl ZmqTransportConfig {
    /// Config with libzmq's default high-water marks (1000), a 100 ms reconnect
    /// interval, a 5 s reconnect cap, and linger `0`.
    ///
    /// Linger `0` keeps teardown non-blocking. Undelivered frames are discarded,
    /// which matches at-most-once delivery.
    ///
    /// ```
    /// use corpus_ipc::ZmqTransportConfig;
    ///
    /// let config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    /// assert_eq!(config.linger_ms, 0);
    /// assert_eq!(config.send_hwm, 1000);
    /// assert_eq!(config.reconnect_ivl_max_ms, 5_000);
    /// ```
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            send_hwm: 1000,
            receive_hwm: 1000,
            linger_ms: 0,
            reconnect_ivl_ms: 100,
            reconnect_ivl_max_ms: 5_000,
        }
    }

    fn check(&self) -> Result<(), ZmqMessageError> {
        self.endpoint_failure()?;
        self.limit_failure()?;
        self.reconnect_failure()
    }

    fn endpoint_failure(&self) -> Result<(), ZmqMessageError> {
        if self.endpoint.is_empty() {
            return Err(invalid_config("endpoint must be non-empty"));
        }
        if self.endpoint.as_bytes().contains(&0) {
            return Err(invalid_config("endpoint must not contain an interior null"));
        }
        Ok(())
    }

    fn limit_failure(&self) -> Result<(), ZmqMessageError> {
        if self.send_hwm < 0 {
            return Err(invalid_config("send_hwm must be >= 0"));
        }
        if self.receive_hwm < 0 {
            return Err(invalid_config("receive_hwm must be >= 0"));
        }
        if self.linger_ms < -1 {
            return Err(invalid_config("linger_ms must be >= -1"));
        }
        Ok(())
    }

    fn reconnect_failure(&self) -> Result<(), ZmqMessageError> {
        if self.reconnect_ivl_ms <= 0 {
            return Err(invalid_config("reconnect_ivl_ms must be > 0"));
        }
        if self.reconnect_ivl_max_ms <= 0 {
            return Err(invalid_config("reconnect_ivl_max_ms must be > 0"));
        }
        if self.reconnect_ivl_max_ms < self.reconnect_ivl_ms {
            return Err(invalid_config(
                "reconnect_ivl_max_ms must be >= reconnect_ivl_ms",
            ));
        }
        Ok(())
    }
}

fn invalid_config(message: &'static str) -> ZmqMessageError {
    ZmqMessageError::InvalidConfig(message)
}

/// Exclusive owner of a libzmq socket.
///
/// Inherits `Send` + `!Sync` from [`zmq::Socket`]. Callers reach it only
/// through the backend mutex, and only with `&mut self`.
struct ExclusiveSocket {
    socket: zmq::Socket,
}

impl ExclusiveSocket {
    fn send(&mut self, bytes: &[u8], flags: i32) -> zmq::Result<()> {
        self.socket.send(bytes, flags)
    }

    fn recv_bytes(&mut self, flags: i32) -> zmq::Result<Vec<u8>> {
        self.socket.recv_bytes(flags)
    }

    fn rcvmore(&mut self) -> zmq::Result<bool> {
        self.socket.get_rcvmore()
    }

    fn option_snapshot(&self) -> zmq::Result<(i32, i32, i32, i32, i32)> {
        Ok((
            self.socket.get_sndhwm()?,
            self.socket.get_rcvhwm()?,
            self.socket.get_linger()?,
            self.socket.get_reconnect_ivl()?,
            self.socket.get_reconnect_ivl_max()?,
        ))
    }
}

enum SocketMode {
    Bind,
    Connect,
}

fn apply_options(socket: &zmq::Socket, config: &ZmqTransportConfig) -> Result<(), ZmqMessageError> {
    socket
        .set_sndhwm(config.send_hwm)
        .map_err(|err| ZmqMessageError::Setup(format!("sndhwm: {err}")))?;
    socket
        .set_rcvhwm(config.receive_hwm)
        .map_err(|err| ZmqMessageError::Setup(format!("rcvhwm: {err}")))?;
    socket
        .set_linger(config.linger_ms)
        .map_err(|err| ZmqMessageError::Setup(format!("linger: {err}")))?;
    socket
        .set_reconnect_ivl(config.reconnect_ivl_ms)
        .map_err(|err| ZmqMessageError::Setup(format!("reconnect_ivl: {err}")))?;
    socket
        .set_reconnect_ivl_max(config.reconnect_ivl_max_ms)
        .map_err(|err| ZmqMessageError::Setup(format!("reconnect_ivl_max: {err}")))?;
    Ok(())
}

fn open_socket(
    context: &zmq::Context,
    kind: zmq::SocketType,
    config: &ZmqTransportConfig,
    mode: SocketMode,
) -> Result<ExclusiveSocket, ZmqMessageError> {
    config.check()?;
    let socket = context
        .socket(kind)
        .map_err(|err| ZmqMessageError::Setup(format!("socket: {err}")))?;
    // Reconnect and queue options must be set before bind/connect. libzmq
    // reads them when it establishes the socket, not on a later setsockopt.
    apply_options(&socket, config)?;
    match &mode {
        SocketMode::Bind => socket
            .bind(&config.endpoint)
            .map_err(|err| ZmqMessageError::Setup(format!("bind {}: {err}", config.endpoint)))?,
        SocketMode::Connect => {
            socket
                .set_subscribe(b"")
                .map_err(|err| ZmqMessageError::Setup(format!("subscribe: {err}")))?;
            socket.connect(&config.endpoint).map_err(|err| {
                ZmqMessageError::Setup(format!("connect {}: {err}", config.endpoint))
            })?;
        }
    }
    Ok(ExclusiveSocket { socket })
}

fn lock_socket(slot: &Mutex<Option<ExclusiveSocket>>) -> MutexGuard<'_, Option<ExclusiveSocket>> {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn map_send(result: zmq::Result<()>) -> Result<(), ZmqMessageError> {
    match result {
        Ok(()) => Ok(()),
        Err(zmq::Error::EAGAIN) => Err(ZmqMessageError::HighWaterMark),
        Err(err) => Err(ZmqMessageError::Transport(err.to_string())),
    }
}

fn decode_frame(bytes: &[u8]) -> Result<IpcMessage, ZmqMessageError> {
    let message = decode_ipc_message_json(bytes)?;
    message.validate()?;
    Ok(message)
}

impl std::fmt::Debug for ZmqMessagePublisher {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ZmqMessagePublisher")
    }
}

/// Publishes one canonical JSON [`IpcMessage`] per ZeroMQ message.
///
/// The socket is a `PUB` socket that binds [`ZmqTransportConfig::endpoint`].
/// `send` validates the payload, then calls [`encode_canonical_ipc_message`],
/// then writes a single frame with `ZMQ_DONTWAIT`.
///
/// See the [module docs](crate::zmq_message) for slow-joiner loss, high-water-mark loss, and
/// linger. `Ok` from [`Self::send`] is acceptance by the local socket, not
/// delivery.
///
/// [`Drop`] clears the socket before `context` is released, so `zmq_close`
/// runs while the ZeroMQ context is still alive.
pub struct ZmqMessagePublisher {
    socket: Mutex<Option<ExclusiveSocket>>,
    /// Kept so `zmq_ctx_term` cannot run before [`Drop`] closes `socket`.
    #[allow(dead_code)]
    context: zmq::Context,
}

impl ZmqMessagePublisher {
    /// Bind a PUB socket. Options from `config` are applied before `bind`.
    pub fn bind(config: ZmqTransportConfig) -> Result<Self, ZmqMessageError> {
        let context = zmq::Context::new();
        let socket = open_socket(&context, zmq::PUB, &config, SocketMode::Bind)?;
        Ok(Self {
            socket: Mutex::new(Some(socket)),
            context,
        })
    }

    /// Validate `message`, encode one canonical envelope, and send it as one frame.
    pub fn send(&self, message: &IpcMessage) -> Result<(), ZmqMessageError> {
        message.validate()?;
        let bytes = encode_canonical_ipc_message(message).map_err(ZmqMessageError::Encode)?;
        self.send_bytes(&bytes)
    }

    fn send_bytes(&self, bytes: &[u8]) -> Result<(), ZmqMessageError> {
        let mut guard = lock_socket(&self.socket);
        let socket = guard.as_mut().ok_or(ZmqMessageError::Closed)?;
        map_send(socket.send(bytes, zmq::DONTWAIT))
    }

    /// Drop the socket. Idempotent. Further [`Self::send`] calls return [`ZmqMessageError::Closed`].
    pub fn close(&self) {
        *lock_socket(&self.socket) = None;
    }

    /// Live socket options after bind: sndhwm, rcvhwm, linger, reconnect interval, reconnect max.
    #[doc(hidden)]
    pub fn socket_options_for_tests(&self) -> Result<(i32, i32, i32, i32, i32), ZmqMessageError> {
        let guard = lock_socket(&self.socket);
        let socket = guard.as_ref().ok_or(ZmqMessageError::Closed)?;
        socket
            .option_snapshot()
            .map_err(|err| ZmqMessageError::Transport(err.to_string()))
    }
}

impl Drop for ZmqMessagePublisher {
    fn drop(&mut self) {
        // Close the socket (`zmq_close`) before the context field is released.
        *lock_socket(&self.socket) = None;
    }
}

impl std::fmt::Debug for ZmqMessageSubscriber {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ZmqMessageSubscriber")
    }
}

/// Subscribes to one canonical JSON [`IpcMessage`] per ZeroMQ message.
///
/// The socket is a `SUB` socket that connects to [`ZmqTransportConfig::endpoint`]
/// and subscribes to every frame. [`Self::try_recv`] is non-blocking.
///
/// [`Drop`] clears the socket before `context` is released, so `zmq_close`
/// runs while the ZeroMQ context is still alive.
pub struct ZmqMessageSubscriber {
    socket: Mutex<Option<ExclusiveSocket>>,
    /// Kept so `zmq_ctx_term` cannot run before [`Drop`] closes `socket`.
    #[allow(dead_code)]
    context: zmq::Context,
}

impl ZmqMessageSubscriber {
    /// Connect a SUB socket and subscribe to the full stream.
    ///
    /// Options from `config` are applied before `connect`. The subscription is
    /// installed before `connect` so libzmq replays it on reconnect.
    pub fn connect(config: ZmqTransportConfig) -> Result<Self, ZmqMessageError> {
        let context = zmq::Context::new();
        let socket = open_socket(&context, zmq::SUB, &config, SocketMode::Connect)?;
        Ok(Self {
            socket: Mutex::new(Some(socket)),
            context,
        })
    }

    /// Receive one frame and decode it with the canonical validation policy.
    ///
    /// [`ZmqMessageError::WouldBlock`] means `EAGAIN` (no message). Malformed
    /// JSON, an incompatible `wire_version`, an invalid payload, and multipart
    /// or empty framing return typed errors. A failed frame is consumed.
    pub fn try_recv(&self) -> Result<IpcMessage, ZmqMessageError> {
        let bytes = self.recv_frame()?;
        decode_frame(&bytes)
    }

    fn recv_frame(&self) -> Result<Vec<u8>, ZmqMessageError> {
        let mut guard = lock_socket(&self.socket);
        let socket = guard.as_mut().ok_or(ZmqMessageError::Closed)?;
        let bytes = match socket.recv_bytes(zmq::DONTWAIT) {
            Ok(bytes) => bytes,
            Err(zmq::Error::EAGAIN) => return Err(ZmqMessageError::WouldBlock),
            Err(err) => return Err(ZmqMessageError::Transport(err.to_string())),
        };
        if socket
            .rcvmore()
            .map_err(|err| ZmqMessageError::Transport(err.to_string()))?
        {
            drain_multipart(socket)?;
            return Err(ZmqMessageError::UnexpectedFraming(FramingFault::Multipart));
        }
        if bytes.is_empty() {
            return Err(ZmqMessageError::UnexpectedFraming(FramingFault::Empty));
        }
        Ok(bytes)
    }

    /// Drop the socket. Idempotent. Further [`Self::try_recv`] calls return [`ZmqMessageError::Closed`].
    pub fn close(&self) {
        *lock_socket(&self.socket) = None;
    }

    /// Live socket options after connect. See [`ZmqMessagePublisher::socket_options_for_tests`].
    #[doc(hidden)]
    pub fn socket_options_for_tests(&self) -> Result<(i32, i32, i32, i32, i32), ZmqMessageError> {
        let guard = lock_socket(&self.socket);
        let socket = guard.as_ref().ok_or(ZmqMessageError::Closed)?;
        socket
            .option_snapshot()
            .map_err(|err| ZmqMessageError::Transport(err.to_string()))
    }
}

impl Drop for ZmqMessageSubscriber {
    fn drop(&mut self) {
        *lock_socket(&self.socket) = None;
    }
}

fn drain_multipart(socket: &mut ExclusiveSocket) -> Result<(), ZmqMessageError> {
    loop {
        match socket.recv_bytes(zmq::DONTWAIT) {
            Ok(_) => {}
            Err(zmq::Error::EAGAIN) => return Ok(()),
            Err(err) => return Err(ZmqMessageError::Transport(err.to_string())),
        }
        if !socket
            .rcvmore()
            .map_err(|err| ZmqMessageError::Transport(err.to_string()))?
        {
            return Ok(());
        }
    }
}

pub(super) fn map_transport_error(err: ZmqMessageError) -> crate::BackendError {
    match err {
        ZmqMessageError::Validation(err) => crate::BackendError::InvalidInput(err.to_string()),
        ZmqMessageError::Encode(CanonicalEncodeError::Validation(err)) => {
            crate::BackendError::InvalidInput(err.to_string())
        }
        other => crate::BackendError::CommunicationError(other.to_string()),
    }
}

// Compile-time Send/Sync record. The raw socket stays `!Sync`; no `unsafe impl Sync`.
const _: fn() = assert_socket_send_sync;

fn assert_socket_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<zmq::Socket>();
    assert_send::<ExclusiveSocket>();
    assert_send::<ZmqMessagePublisher>();
    assert_sync::<ZmqMessagePublisher>();
    assert_send::<ZmqMessageSubscriber>();
    assert_sync::<ZmqMessageSubscriber>();

    <zmq::Socket as not_sync::IfSyncThenAmbiguous<_>>::witness();
    <ExclusiveSocket as not_sync::IfSyncThenAmbiguous<_>>::witness();
}

/// Overlapping-impl witness that a concrete type is not `Sync`.
mod not_sync {
    pub trait IfSyncThenAmbiguous<Marker> {
        fn witness() {}
    }

    pub struct NotSyncToken;
    pub struct SyncToken;

    impl<T: ?Sized> IfSyncThenAmbiguous<NotSyncToken> for T {}
    impl<T: ?Sized + Sync> IfSyncThenAmbiguous<SyncToken> for T {}
}

#[cfg(test)]
mod coverage {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::ValidationError;

    fn endpoint() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "ipc:///tmp/corpus_ipc_zmq_cov_{}_{}.ipc",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn assert_invalid(config: &ZmqTransportConfig, expected: &str) {
        match config.check() {
            Err(ZmqMessageError::InvalidConfig(message)) => assert_eq!(message, expected),
            other => panic!("expected invalid config, got {other:?}"),
        }
    }

    #[test]
    fn config_check_rejects_each_constraint() {
        assert_invalid(&ZmqTransportConfig::new(""), "endpoint must be non-empty");
        assert_invalid(
            &ZmqTransportConfig::new("ipc://bad\0endpoint"),
            "endpoint must not contain an interior null",
        );

        let mut config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.send_hwm = -1;
        assert_invalid(&config, "send_hwm must be >= 0");

        config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.receive_hwm = -1;
        assert_invalid(&config, "receive_hwm must be >= 0");

        config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.linger_ms = -2;
        assert_invalid(&config, "linger_ms must be >= -1");

        config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.reconnect_ivl_ms = 0;
        assert_invalid(&config, "reconnect_ivl_ms must be > 0");

        config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.reconnect_ivl_max_ms = 0;
        assert_invalid(&config, "reconnect_ivl_max_ms must be > 0");

        config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
        config.reconnect_ivl_ms = 200;
        config.reconnect_ivl_max_ms = 100;
        assert_invalid(&config, "reconnect_ivl_max_ms must be >= reconnect_ivl_ms");

        ZmqTransportConfig::new("tcp://127.0.0.1:9")
            .check()
            .unwrap();
    }

    #[test]
    fn send_and_transport_errors_map_without_a_socket() {
        assert!(map_send(Ok(())).is_ok());
        assert!(matches!(
            map_send(Err(zmq::Error::EAGAIN)),
            Err(ZmqMessageError::HighWaterMark)
        ));
        assert!(matches!(
            map_send(Err(zmq::Error::EINVAL)),
            Err(ZmqMessageError::Transport(_))
        ));

        let validation = ValidationError::non_finite("strength", f32::NAN);
        assert!(matches!(
            map_transport_error(ZmqMessageError::Validation(validation.clone())),
            crate::BackendError::InvalidInput(_)
        ));
        assert!(matches!(
            map_transport_error(ZmqMessageError::Encode(CanonicalEncodeError::Validation(
                validation
            ))),
            crate::BackendError::InvalidInput(_)
        ));
        assert!(matches!(
            map_transport_error(ZmqMessageError::WouldBlock),
            crate::BackendError::CommunicationError(_)
        ));
    }

    #[test]
    fn publisher_and_subscriber_debug_and_closed_options() {
        assert_socket_send_sync();
        let endpoint = endpoint();
        let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
        let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
        assert_eq!(format!("{publisher:?}"), "ZmqMessagePublisher");
        assert_eq!(format!("{subscriber:?}"), "ZmqMessageSubscriber");
        publisher.close();
        subscriber.close();
        assert!(matches!(
            publisher.socket_options_for_tests(),
            Err(ZmqMessageError::Closed)
        ));
        assert!(matches!(
            subscriber.socket_options_for_tests(),
            Err(ZmqMessageError::Closed)
        ));

        poison_publisher_lock(&publisher);
        assert!(matches!(
            publisher.send(&crate::IpcMessage::Ping),
            Err(ZmqMessageError::Closed)
        ));
    }

    fn poison_publisher_lock(publisher: &ZmqMessagePublisher) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = publisher.socket.lock().expect("lock");
            panic!("poison the publisher mutex");
        }));
    }

    #[test]
    fn bind_and_connect_report_setup_failures() {
        let bind_err =
            ZmqMessagePublisher::bind(ZmqTransportConfig::new("not-an-endpoint")).unwrap_err();
        assert!(bind_err.to_string().contains("bind"), "{bind_err}");

        let connect_err =
            ZmqMessageSubscriber::connect(ZmqTransportConfig::new("not-an-endpoint")).unwrap_err();
        assert!(connect_err.to_string().contains("connect"), "{connect_err}");
    }
}
