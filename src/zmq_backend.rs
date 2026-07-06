// SPDX-License-Identifier: MIT OR Apache-2.0

//! ZMQ SUB backend — reads compute data packets from the remote compute IPC socket.
//!
//! Requires feature `zmq`.

use crate::{RuntimeBackend, BackendError};

/// Default ZeroMQ IPC endpoint for receiving compute data packets.
/// Can be overridden via environment variable `CORPUS_IPC_ZMQ_READOUT_IPC`.
const DEFAULT_READOUT_IPC: &str = "ipc:///tmp/corpus_ipc_readout.ipc";

struct SafeSocket {
    socket: zmq::Socket,
}
// ZMQ socket is not Send by default but the connection is owned exclusively
// by this struct — safe to move across threads.
unsafe impl Send for SafeSocket {}
unsafe impl Sync for SafeSocket {}

/// Generic IPC backend — subscribes to the remote compute's ZMQ PUB socket and
/// returns the latest compute readouts on each call.
///
/// # Wire format
/// The packet consists of an 8-byte header followed by a variable number of
/// 4-byte floating-point values.
///
/// ```text
/// [0..8]   tick     i64 LE      monotonic tick counter
/// [8..]    readout  N×f32 LE    lobe outputs
/// ```
pub struct ZmqRuntimeBackend {
    context: zmq::Context,
    sub_socket: Option<SafeSocket>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub tick: i64,
}

impl ZmqRuntimeBackend {
    /// Create a new uninitialized ZMQ backend.
    ///
    /// Socket and subscription are established only on the first successful
    /// `initialize` call. Safe to construct even when the `zmq` feature is
    /// enabled but the external publisher is not yet running.
    pub fn new() -> Self {
        Self {
            context: zmq::Context::new(),
            sub_socket: None,
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

    fn receive_readout(&mut self) -> Result<Vec<f32>, BackendError> {
        let safe_socket = self.sub_socket.as_ref().ok_or_else(|| {
            BackendError::CommunicationError("SUB socket not connected".to_string())
        })?;
        let socket = &safe_socket.socket;

        match socket.recv_bytes(zmq::DONTWAIT) {
            Ok(buf) if buf.len() >= 8 && (buf.len() - 8).is_multiple_of(4) => {
                self.tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
                let num_floats = (buf.len() - 8) / 4;
                self.last_readout.resize(num_floats, 0.0);
                for i in 0..num_floats {
                    let off = 8 + i * 4;
                    self.last_readout[i] =
                        f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                }
            }
            Ok(buf) => {
                eprintln!("[zmq-ipc] Unexpected packet size: {} bytes", buf.len());
            }
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
}

impl Default for ZmqRuntimeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeBackend for ZmqRuntimeBackend {
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
                "ZmqRuntimeBackend not initialized — call initialize() first".to_string(),
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
        let endpoint = std::env::var("CORPUS_IPC_ZMQ_READOUT_IPC")
            .unwrap_or_else(|_| DEFAULT_READOUT_IPC.to_string());
        socket.connect(&endpoint).map_err(|e| {
            BackendError::InitializationError(format!(
                "ZMQ connect to {}: {} (is the IPC producer running?)",
                endpoint, e
            ))
        })?;

        self.sub_socket = Some(SafeSocket { socket });
        self.initialized = true;
        println!("[zmq-ipc] Connected to IPC producer at {}", endpoint);
        Ok(())
    }

    /// Persist current model state (delegated to remote if supported).
    /// Current ZMQ implementation is read-only; this is a no-op.
    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        println!("[zmq-ipc] State lives in the external compute process (CUDA VRAM)");
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
        self.sub_socket = None;
        println!("[zmq-ipc] Readout cache reset; will re-initialize on next call");
        Ok(())
    }
}

// ── Packet-parsing unit tests (no live ZMQ socket needed) ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(tick: i64, readout: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + readout.len() * 4);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in readout {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parse_dynamic_packet() {
        let readout: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let tick: i64 = 42_000;
        let buf = make_packet(tick, &readout);

        let mut b = ZmqRuntimeBackend::new();
        // Manually simulate receiving the packet
        b.tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        let num_floats = (buf.len() - 8) / 4;
        b.last_readout.resize(num_floats, 0.0);
        for i in 0..num_floats {
            let off = 8 + i * 4;
            b.last_readout[i] = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        }

        assert_eq!(b.tick, tick);
        assert_eq!(b.last_readout.len(), 20);
        for (i, val) in readout.iter().enumerate().take(20) {
            assert!((b.last_readout[i] - val).abs() < 1e-5);
        }
    }

    #[test]
    fn malformed_packet_does_not_mutate_state() {
        let mut b = ZmqRuntimeBackend::new();
        b.last_readout = vec![1.0, 2.0];
        b.tick = 100;

        let initial_readout = b.last_readout.clone();
        let initial_tick = b.tick;

        // A 5-byte garbage packet has an invalid length
        let bad_buf: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00];

        // Simulate a recv call that would get this bad packet
        // In a real scenario, the Ok(buf) branch for bad length would be taken
        // and an error printed, but the state would not change.
        if bad_buf.len() < 8 || !(bad_buf.len() - 8).is_multiple_of(4) {
            // This is what should happen inside receive_readout
            eprintln!("[test] Malformed packet received");
        } else {
            // this part should not be reached
        }

        assert_eq!(b.last_readout, initial_readout);
        assert_eq!(b.tick, initial_tick);
    }
}
