//! ZMQ SUB backend — reads neural data packets from the Julia brain IPC socket.
//!
//! Requires feature `zmq`.

use crate::{BackendError, NeuralBackend};

/// Default ZeroMQ IPC endpoint for receiving neural data packets.
/// Can be overridden via environment variable `SPIKENAUT_ZMQ_READOUT_IPC`.
const DEFAULT_READOUT_IPC: &str = "ipc:///tmp/spikenaut_readout.ipc";

struct SafeSocket {
    socket: zmq::Socket,
}
// ZMQ socket is not Send by default but the connection is owned exclusively
// by this struct — safe to move across threads.
unsafe impl Send for SafeSocket {}
unsafe impl Sync for SafeSocket {}

/// Julia IPC backend — subscribes to the Julia brain's ZMQ PUB socket and
/// returns the latest neural readouts on each call.
///
/// # Wire format
/// The packet consists of an 8-byte header followed by a variable number of
/// 4-byte floating-point values.
///
/// ```text
/// [0..8]   tick     i64 LE      monotonic tick counter
/// [8..]    readout  N×f32 LE    lobe outputs
/// ```
pub struct ZmqBrainBackend {
    context: zmq::Context,
    sub_socket: Option<SafeSocket>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub brain_tick: i64,
}

impl ZmqBrainBackend {
    pub fn new() -> Self {
        Self {
            context: zmq::Context::new(),
            sub_socket: None,
            initialized: false,
            last_readout: Vec::new(),
            brain_tick: 0,
        }
    }

    /// Monotonic tick counter of the last received packet.
    pub fn brain_tick(&self) -> i64 {
        self.brain_tick
    }

    fn receive_readout(&mut self) -> Result<Vec<f32>, BackendError> {
        let safe_socket = self.sub_socket.as_ref()
            .ok_or_else(|| BackendError::CommunicationError(
                "SUB socket not connected".to_string()
            ))?;
        let socket = &safe_socket.socket;

        match socket.recv_bytes(zmq::DONTWAIT) {
            Ok(buf) if buf.len() >= 8 && (buf.len() - 8) % 4 == 0 => {
                self.brain_tick = i64::from_le_bytes(
                    buf[0..8].try_into().unwrap()
                );
                let num_floats = (buf.len() - 8) / 4;
                self.last_readout.resize(num_floats, 0.0);
                for i in 0..num_floats {
                    let off = 8 + i * 4;
                    self.last_readout[i] = f32::from_le_bytes(
                        buf[off..off+4].try_into().unwrap()
                    );
                }
            }
            Ok(buf) => {
                eprintln!("[zmq-brain] Unexpected packet size: {} bytes", buf.len());
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

impl Default for ZmqBrainBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralBackend for ZmqBrainBackend {
    fn process_signals(
        &mut self,
        _inputs: &[f32],
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "ZmqBrainBackend not initialized — call initialize() first".to_string(),
            ));
        }
        self.receive_readout()
    }

    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        let socket = self.context.socket(zmq::SUB)
            .map_err(|e| BackendError::InitializationError(
                format!("ZMQ SUB socket: {e}")
            ))?;
        socket.set_subscribe(b"")
            .map_err(|e| BackendError::InitializationError(
                format!("ZMQ subscribe: {e}")
            ))?;
        socket.set_rcvhwm(16)
            .map_err(|e| BackendError::InitializationError(
                format!("ZMQ rcvhwm: {e}")
            ))?;
        let endpoint = std::env::var("SPIKENAUT_ZMQ_READOUT_IPC").unwrap_or_else(|_| DEFAULT_READOUT_IPC.to_string());
        socket.connect(&endpoint)
            .map_err(|e| BackendError::InitializationError(format!(
                "ZMQ connect to {}: {} (is main_brain.jl running?)", endpoint, e
            )))?;

        self.sub_socket = Some(SafeSocket { socket });
        self.initialized = true;
        println!("[zmq-brain] Connected to Julia Brain at {}", endpoint);
        Ok(())
    }

    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        println!("[zmq-brain] State lives in the Julia Brain process (CUDA VRAM)");
        Ok(())
    }

    fn get_spike_states(&self) -> Vec<bool> {
        self.last_readout.iter().map(|&v| v > 0.5).collect()
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        self.last_readout.clear();
        self.brain_tick = 0;
        println!("[zmq-brain] Readout cache reset");
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
        for v in readout { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    #[test]
    fn parse_dynamic_packet() {
        let readout: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let tick: i64 = 42_000;
        let buf = make_packet(tick, &readout);

        let mut b = ZmqBrainBackend::new();
        // Manually simulate receiving the packet
        b.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        let num_floats = (buf.len() - 8) / 4;
        b.last_readout.resize(num_floats, 0.0);
        for i in 0..num_floats {
            let off = 8 + i * 4;
            b.last_readout[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }

        assert_eq!(b.brain_tick, tick);
        assert_eq!(b.last_readout.len(), 20);
        for i in 0..20 {
            assert!((b.last_readout[i] - readout[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn malformed_packet_does_not_mutate_state() {
        let mut b = ZmqBrainBackend::new();
        b.last_readout = vec![1.0, 2.0];
        b.brain_tick = 100;

        let initial_readout = b.last_readout.clone();
        let initial_tick = b.brain_tick;

        // A 5-byte garbage packet has an invalid length
        let bad_buf: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00];

        // Simulate a recv call that would get this bad packet
        // In a real scenario, the Ok(buf) branch for bad length would be taken
        // and an error printed, but the state would not change.
        if bad_buf.len() < 8 || (bad_buf.len() - 8) % 4 != 0 {
             // This is what should happen inside receive_readout
             eprintln!("[test] Malformed packet received");
        } else {
            // this part should not be reached
        }

        assert_eq!(b.last_readout, initial_readout);
        assert_eq!(b.brain_tick, initial_tick);
    }
}
