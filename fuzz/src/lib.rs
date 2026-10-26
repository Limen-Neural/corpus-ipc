// SPDX-License-Identifier: MIT OR Apache-2.0

//! Invariants checked by the cargo-fuzz targets (LIM-1277).
//!
//! Every check drives a production entry point; nothing here re-implements a
//! parser. The same file is compiled into `tests/fuzz_regressions.rs`, so
//! committed regression seeds replay on stable `cargo test --all-features`
//! without libFuzzer. Keep it dependent on `corpus_ipc` and `std` only.

use std::sync::{Mutex, OnceLock};

use corpus_ipc::zmq_readout::{DEFAULT_MAX_READOUT_FLOATS, parse_readout_packet};
use corpus_ipc::{
    FramingFault, IpcMessage, STRUCTURED_MAX_FRAME_BYTES, Validate, ZmqIpcBackend, ZmqMessageError,
    ZmqMessageSubscriber, decode_ipc_message_json, encode_canonical_ipc_message,
};

/// Small readout cap so short fuzz inputs reach the over-limit branch.
const SMALL_READOUT_FLOATS: usize = 16;

/// `fuzz_ipc_message_json`: arbitrary bytes through the compatibility decoder.
///
/// Accepted messages must pass [`Validate`] at `ProtocolLimits::DEFAULT` and
/// round-trip byte-stably through the canonical encoder.
pub fn check_ipc_message_json(data: &[u8]) {
    match decode_ipc_message_json(data) {
        Ok(message) => {
            message
                .validate()
                .expect("decoder accepted a message that fails ProtocolLimits::DEFAULT");
            assert_canonical_round_trip(&message, |bytes| {
                decode_ipc_message_json(bytes).map_err(|err| err.to_string())
            });
        }
        Err(err) => {
            let _ = err.to_string();
        }
    }
}

/// `fuzz_zmq_readout_frame`: arbitrary bytes through `parse_readout_packet`
/// and the `ZmqIpcBackend` ingest path built on it.
pub fn check_zmq_readout_frame(data: &[u8]) {
    for limit in [DEFAULT_MAX_READOUT_FLOATS, SMALL_READOUT_FLOATS] {
        check_readout_against_layout(data, limit);
        check_backend_ingest(data, limit);
    }
}

/// `fuzz_zmq_message_frame`: one arbitrary frame through the decoder
/// `ZmqMessageSubscriber::try_recv` runs.
///
/// Empty frames are framing errors. Every other rejection is an envelope or
/// validation error, never a transport error. Accepted messages agree with
/// [`decode_ipc_message_json`], pass [`Validate`], and round-trip through the
/// canonical encoder within [`STRUCTURED_MAX_FRAME_BYTES`].
pub fn check_zmq_message_frame(data: &[u8]) {
    let frame = ZmqMessageSubscriber::decode_frame_for_tests(data);
    if data.is_empty() {
        assert!(
            matches!(
                frame,
                Err(ZmqMessageError::UnexpectedFraming(FramingFault::Empty))
            ),
            "empty frame must be a framing error, got {frame:?}"
        );
        return;
    }
    match frame {
        Ok(message) => {
            let compat = decode_ipc_message_json(data)
                .expect("frame decoder accepted bytes the compatibility decoder rejects");
            assert_eq!(compat, message, "frame and compatibility decoders disagree");
            message
                .validate()
                .expect("frame decoder accepted a message that fails Validate");
            let encoded = assert_canonical_round_trip(&message, |bytes| {
                ZmqMessageSubscriber::decode_frame_for_tests(bytes).map_err(|err| err.to_string())
            });
            assert!(
                i64::try_from(encoded.len()).is_ok_and(|len| len <= STRUCTURED_MAX_FRAME_BYTES),
                "canonical frame exceeds STRUCTURED_MAX_FRAME_BYTES"
            );
        }
        Err(ZmqMessageError::Envelope(_) | ZmqMessageError::Validation(_)) => {}
        Err(other) => panic!("non-empty frame produced a non-decode error: {other:?}"),
    }
}

/// Encode canonically, decode with `decode`, and require the same message and
/// identical bytes on re-encode. Returns the canonical bytes.
fn assert_canonical_round_trip(
    message: &IpcMessage,
    decode: impl Fn(&[u8]) -> Result<IpcMessage, String>,
) -> Vec<u8> {
    let encoded =
        encode_canonical_ipc_message(message).expect("validated message must encode canonically");
    let decoded = decode(&encoded).expect("canonical bytes must decode");
    assert_eq!(
        &decoded, message,
        "canonical round trip changed the message"
    );
    let reencoded = encode_canonical_ipc_message(&decoded).expect("decoded message must re-encode");
    assert_eq!(reencoded, encoded, "canonical encoding is not stable");
    encoded
}

fn check_readout_against_layout(data: &[u8], limit: usize) {
    let parsed = parse_readout_packet(data, limit);
    let Some(payload) = data.len().checked_sub(8) else {
        assert!(parsed.is_err(), "short packet accepted");
        return;
    };
    if payload % 4 != 0 || payload / 4 > limit {
        assert!(parsed.is_err(), "misaligned or over-limit packet accepted");
        return;
    }
    let (tick, readout) = parsed.expect("well-formed packet rejected");
    assert_eq!(tick, i64::from_le_bytes(data[..8].try_into().unwrap()));
    assert_eq!(readout.len(), payload / 4);
    for (value, chunk) in readout.iter().zip(data[8..].chunks_exact(4)) {
        assert_eq!(
            value.to_bits(),
            u32::from_le_bytes(chunk.try_into().unwrap())
        );
    }
}

/// Tick-only packet the backend is reset to before every ingest check.
const SENTINEL_PACKET: [u8; 8] = i64::MIN.to_le_bytes();

/// The backend cache changes exactly when the parser accepts the packet.
///
/// The backend is reset to [`SENTINEL_PACKET`] first, so a rejected packet
/// that wrongly wrote its own data cannot hide behind an earlier accept of
/// the same bytes at a larger limit.
fn check_backend_ingest(data: &[u8], limit: usize) {
    static BACKEND: OnceLock<Mutex<ZmqIpcBackend>> = OnceLock::new();
    let mut backend = BACKEND
        .get_or_init(|| Mutex::new(ZmqIpcBackend::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    backend
        .apply_readout_packet_with_limit_for_tests(&SENTINEL_PACKET, 0)
        .expect("sentinel packet must be accepted");
    let before = backend.readout_cache_snapshot_for_tests();
    let expected = parse_readout_packet(data, limit);
    let applied = backend.apply_readout_packet_with_limit_for_tests(data, limit);
    let after = backend.readout_cache_snapshot_for_tests();
    match expected {
        Ok((tick, readout)) => {
            applied.expect("backend rejected a packet the parser accepts");
            assert_eq!(after.0, tick);
            assert_eq!(bits(&after.1), bits(&readout));
        }
        Err(_) => {
            assert!(
                applied.is_err(),
                "backend accepted a packet the parser rejects"
            );
            assert_eq!((after.0, bits(&after.1)), (before.0, bits(&before.1)));
        }
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
