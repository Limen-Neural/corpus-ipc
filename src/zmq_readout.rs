// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure parser for ZMQ SUB readout packets (`i64` tick + `N × f32` LE payload).
//!
//! Used by [`crate::ZmqIpcBackend`] and conformance tests. This layer bounds the
//! **decoded** float vector only; libzmq still allocates the raw `recv_bytes`
//! buffer before parsing runs.

use crate::BackendError;

/// Default maximum number of `f32` readout values after the 8-byte tick header.
///
/// Justification: production deployments documented in this crate use on the order
/// of 16 lobe readouts (72-byte payload) or 20 scalar slots when an 88-byte
/// frame is treated as tick + 20 floats (see wire docs). `1024` floats (4 KiB
/// payload) is two orders of magnitude above that operational range while
/// capping decoded cache growth if a publisher misbehaves. Larger networks can
/// raise the limit via [`ENV_MAX_READOUT_FLOATS`].
pub const DEFAULT_MAX_READOUT_FLOATS: usize = 1024;

/// Environment variable overriding [`DEFAULT_MAX_READOUT_FLOATS`].
pub const ENV_MAX_READOUT_FLOATS: &str = "CORPUS_IPC_ZMQ_MAX_READOUT_FLOATS";

/// Resolved readout float cap (env override or [`DEFAULT_MAX_READOUT_FLOATS`]).
pub fn max_readout_float_limit() -> usize {
    std::env::var(ENV_MAX_READOUT_FLOATS)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_READOUT_FLOATS)
}

/// Parse a tick + readout packet without mutating backend state.
///
/// # Errors
/// - [`BackendError::CommunicationError`] — truncated header or misaligned payload length.
/// - [`BackendError::InvalidInput`] — float count above `max_floats`.
pub fn parse_readout_packet(
    buf: &[u8],
    max_floats: usize,
) -> Result<(i64, Vec<f32>), BackendError> {
    if buf.len() < 8 {
        return Err(BackendError::CommunicationError(format!(
            "ZMQ readout packet too short: {} bytes (need 8-byte tick header)",
            buf.len()
        )));
    }
    let payload_len = buf.len() - 8;
    if !payload_len.is_multiple_of(4) {
        return Err(BackendError::CommunicationError(format!(
            "ZMQ readout payload misaligned: {payload_len} bytes after header (must be a multiple of 4)"
        )));
    }
    let num_floats = payload_len / 4;
    if num_floats > max_floats {
        return Err(BackendError::InvalidInput(format!(
            "ZMQ readout float count {num_floats} exceeds limit {max_floats}"
        )));
    }

    let tick = i64::from_le_bytes(buf[0..8].try_into().expect("length checked"));
    let mut readout = Vec::new();
    if num_floats > 0 {
        readout = Vec::with_capacity(num_floats);
        for i in 0..num_floats {
            let off = 8 + i * 4;
            readout.push(f32::from_le_bytes(
                buf[off..off + 4].try_into().expect("aligned payload"),
            ));
        }
    }
    Ok((tick, readout))
}
