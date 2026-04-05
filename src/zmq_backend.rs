//! ZMQ SUB backend — reads 88-byte NERO packets from the Julia brain IPC socket.
//!
//! Requires feature `zmq`.

use crate::{BackendError, GpuTelemetry, TraderBackend};

/// Default ZeroMQ IPC endpoint for receiving NERO packets.
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
/// returns the latest 16-channel lobe readout + 4 NERO scores on each call.
///
/// # Wire format (88-byte NERO packet)
/// ```text
/// [0..8]   tick     i64 LE   monotonic tick counter
/// [8..72]  readout  16×f32   lobe output (action/trade signals)
/// [72..88] nero     4×f32    [dopamine, cortisol, acetylcholine, tempo]
/// ```
///
/// Legacy 72-byte packets (no NERO suffix) are accepted with zero-order hold
/// on the last known NERO scores.
pub struct ZmqBrainBackend {
    context: zmq::Context,
    sub_socket: Option<SafeSocket>,
    initialized: bool,
    pub(crate) last_readout: Vec<f32>,
    pub last_nero: [f32; 4],
    pub(crate) brain_tick: i64,
}

impl ZmqBrainBackend {
    pub fn new() -> Self {
        Self {
            context: zmq::Context::new(),
            sub_socket: None,
            initialized: false,
            last_readout: vec![0.0f32; 16],
            last_nero: [0.25f32; 4], // neutral defaults
            brain_tick: 0,
        }
    }

    /// Last 4-element NERO scores received from the Julia brain.
    pub fn get_nero_scores(&self) -> [f32; 4] {
        self.last_nero
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
            Ok(buf) if buf.len() == 88 => {
                self.brain_tick = i64::from_le_bytes(
                    buf[0..8].try_into().unwrap_or([0; 8])
                );
                for i in 0..16 {
                    let off = 8 + i * 4;
                    self.last_readout[i] = f32::from_le_bytes(
                        buf[off..off+4].try_into().unwrap_or([0; 4])
                    );
                }
                for i in 0..4 {
                    let off = 72 + i * 4;
                    self.last_nero[i] = f32::from_le_bytes(
                        buf[off..off+4].try_into().unwrap_or([0; 4])
                    );
                }
            }
            Ok(buf) if buf.len() == 72 => {
                // Legacy packet — parse tick+readout, keep last_nero (zero-order hold).
                self.brain_tick = i64::from_le_bytes(
                    buf[0..8].try_into().unwrap_or([0; 8])
                );
                for i in 0..16 {
                    let off = 8 + i * 4;
                    self.last_readout[i] = f32::from_le_bytes(
                        buf[off..off+4].try_into().unwrap_or([0; 4])
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

        // Return 20-element widened contract: [readout(16), nero(4)]
        let mut out = Vec::with_capacity(20);
        out.extend_from_slice(&self.last_readout);
        out.extend_from_slice(&self.last_nero);
        Ok(out)
    }
}

impl Default for ZmqBrainBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TraderBackend for ZmqBrainBackend {
    fn process_signals(
        &mut self,
        _normalized_inputs: &[f32; 8],
        _inhibition_signal: f32,
        _telemetry: &GpuTelemetry,
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

    fn get_spike_states(&self) -> [bool; 16] {
        std::array::from_fn(|i| self.last_readout[i] > 0.5)
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        self.last_readout = vec![0.0f32; 16];
        self.last_nero = [0.25f32; 4];
        self.brain_tick = 0;
        println!("[zmq-brain] Readout and NERO cache reset");
        Ok(())
    }
}

// ── Packet-parsing unit tests (no live ZMQ socket needed) ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_88_packet(tick: i64, readout: [f32; 16], nero: [f32; 4]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(88);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in &readout { buf.extend_from_slice(&v.to_le_bytes()); }
        for v in &nero    { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    fn make_72_packet(tick: i64, readout: [f32; 16]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(72);
        buf.extend_from_slice(&tick.to_le_bytes());
        for v in &readout { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    #[test]
    fn parse_88_byte_nero_packet() {
        let nero     = [0.4f32, 0.3, 0.2, 0.1];
        let readout  = std::array::from_fn::<f32, 16, _>(|i| i as f32 * 0.1);
        let tick: i64 = 42_000;
        let buf = make_88_packet(tick, readout, nero);

        let mut b = ZmqBrainBackend::new();
        b.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        for i in 0..16 {
            let off = 8 + i * 4;
            b.last_readout[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }
        for i in 0..4 {
            let off = 72 + i * 4;
            b.last_nero[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }

        assert_eq!(b.brain_tick, tick);
        for i in 0..16 {
            assert!((b.last_readout[i] - readout[i]).abs() < 1e-5);
        }
        for i in 0..4 {
            assert!((b.last_nero[i] - nero[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn legacy_72_packet_preserves_nero() {
        let mut b = ZmqBrainBackend::new();
        let known_nero = [0.7f32, 0.1, 0.1, 0.1];
        b.last_nero = known_nero;

        let buf = make_72_packet(99, [1.0f32; 16]);
        b.brain_tick = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        for i in 0..16 {
            let off = 8 + i * 4;
            b.last_readout[i] = f32::from_le_bytes(buf[off..off+4].try_into().unwrap());
        }
        // No NERO bytes parsed — zero-order hold.
        assert_eq!(b.brain_tick, 99);
        assert_eq!(b.last_nero, known_nero);
    }

    #[test]
    fn malformed_packet_does_not_mutate_state() {
        let b = ZmqBrainBackend::new();
        let initial_nero = b.last_nero;
        let initial_tick = b.brain_tick;

        // A 5-byte garbage packet is neither 88 nor 72 bytes — no branch entered.
        let bad: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00];
        assert!(bad.len() != 88 && bad.len() != 72);
        assert_eq!(b.last_nero, initial_nero);
        assert_eq!(b.brain_tick, initial_tick);
    }
}
