// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stable replay of the cargo-fuzz invariants (LIM-1277).
//!
//! Runs the exact checks the libFuzzer targets run (`fuzz/src/lib.rs`) over the
//! committed regression seeds, the committed compatibility vectors (read in
//! place, never copied), and synthetic empty/truncated/random/oversize inputs.
//! No libFuzzer or nightly toolchain is needed; requires feature `zmq`.

#[path = "../fuzz/src/lib.rs"]
mod checks;

use std::fs;
use std::path::{Path, PathBuf};

use corpus_ipc::zmq_readout::DEFAULT_MAX_READOUT_FLOATS;
use corpus_ipc::{ProtocolLimits, ZmqMessageSubscriber, decode_ipc_message_json};
use serde_json::{Value, json};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Regular files in `dir`, sorted, skipping dotfiles (`.gitkeep`).
fn files_in(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(repo_path(dir))
        .unwrap_or_else(|err| panic!("read {dir}: {err}"))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .filter(|path| !path.file_name().unwrap().to_string_lossy().starts_with('.'))
        .collect();
    files.sort();
    files
}

fn read(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn wire_v1_vectors() -> Vec<Vec<u8>> {
    files_in("test-vectors/wire-v1")
        .iter()
        .map(|path| read(path))
        .collect()
}

fn zmq_readout_vectors() -> Vec<Vec<u8>> {
    files_in("test-vectors/zmq")
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "hex"))
        .map(|path| hex::decode(String::from_utf8(read(path)).unwrap().trim()).unwrap())
        .collect()
}

/// Deterministic pseudo-random inputs (xorshift64), so failures reproduce.
fn random_inputs(count: usize, max_len: usize) -> Vec<Vec<u8>> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..count)
        .map(|_| {
            let len = (next() as usize) % (max_len + 1);
            (0..len).map(|_| next() as u8).collect()
        })
        .collect()
}

/// Every proper prefix of `bytes` (truncations), including empty.
fn prefixes(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    (0..bytes.len()).map(move |len| &bytes[..len])
}

fn spikes_envelope(spike_count: usize) -> Vec<u8> {
    let spike = json!({"channel": 1, "strength": 0.5, "time": 2});
    serde_json::to_vec(&json!({
        "wire_version": 1,
        "payload": {"Spikes": {
            "batch_id": 1,
            "metadata": null,
            "session_id": null,
            "spikes": vec![spike; spike_count],
            "timestamp": 0
        }}
    }))
    .unwrap()
}

fn oversize_session_id() -> Vec<u8> {
    let mut value: Value =
        serde_json::from_slice(&read(&repo_path("test-vectors/wire-v1/spikes.json"))).unwrap();
    value["payload"]["Spikes"]["session_id"] =
        Value::String("s".repeat(ProtocolLimits::DEFAULT.max_string_bytes + 1));
    serde_json::to_vec(&value).unwrap()
}

#[test]
fn ipc_message_json_regressions_hold() {
    for path in files_in("fuzz/regressions/fuzz_ipc_message_json") {
        checks::check_ipc_message_json(&read(&path));
    }
}

#[test]
fn zmq_readout_frame_regressions_hold() {
    for path in files_in("fuzz/regressions/fuzz_zmq_readout_frame") {
        checks::check_zmq_readout_frame(&read(&path));
    }
}

#[test]
fn zmq_message_frame_regressions_hold() {
    for path in files_in("fuzz/regressions/fuzz_zmq_message_frame") {
        checks::check_zmq_message_frame(&read(&path));
    }
}

#[test]
fn json_targets_hold_on_committed_vectors_and_truncations() {
    let vectors = wire_v1_vectors();
    assert!(!vectors.is_empty());
    for vector in &vectors {
        for input in prefixes(vector).chain([vector.as_slice()]) {
            checks::check_ipc_message_json(input);
            checks::check_zmq_message_frame(input);
        }
    }
}

#[test]
fn readout_target_holds_on_committed_frames_and_truncations() {
    let frames = zmq_readout_vectors();
    assert!(!frames.is_empty());
    for frame in &frames {
        for input in prefixes(frame).chain([frame.as_slice()]) {
            checks::check_zmq_readout_frame(input);
        }
    }
}

#[test]
fn all_targets_hold_on_empty_and_random_bytes() {
    let mut inputs = random_inputs(512, 256);
    inputs.push(Vec::new());
    for input in &inputs {
        checks::check_ipc_message_json(input);
        checks::check_zmq_message_frame(input);
        checks::check_zmq_readout_frame(input);
    }
}

#[test]
fn oversize_json_inputs_are_rejected_not_accepted() {
    let max = ProtocolLimits::DEFAULT.max_spike_events;
    let at_limit = spikes_envelope(max);
    assert!(decode_ipc_message_json(&at_limit).is_ok());
    for input in [at_limit, spikes_envelope(max + 1), oversize_session_id()] {
        checks::check_ipc_message_json(&input);
        checks::check_zmq_message_frame(&input);
    }
    assert!(decode_ipc_message_json(&spikes_envelope(max + 1)).is_err());
    assert!(ZmqMessageSubscriber::decode_frame_for_tests(&spikes_envelope(max + 1)).is_err());
    assert!(decode_ipc_message_json(&oversize_session_id()).is_err());
}

#[test]
fn oversize_readout_frame_is_rejected() {
    let over = vec![0u8; 8 + 4 * (DEFAULT_MAX_READOUT_FLOATS + 1)];
    assert!(
        corpus_ipc::zmq_readout::parse_readout_packet(&over, DEFAULT_MAX_READOUT_FLOATS).is_err()
    );
    checks::check_zmq_readout_frame(&over);
}
