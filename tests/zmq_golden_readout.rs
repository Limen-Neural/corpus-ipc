// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden ZMQ readout frames (LIM-1275 / LIM-1328).

use std::path::PathBuf;

use corpus_ipc::zmq_readout::{DEFAULT_MAX_READOUT_FLOATS, parse_readout_packet};
use corpus_ipc::{BackendError, ZmqIpcBackend};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    name: String,
    file: String,
    sha256: String,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/zmq")
}

fn load_fixture_bytes(entry: &ManifestEntry) -> Vec<u8> {
    let path = fixture_root().join(&entry.file);
    let hex =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    hex::decode(hex.trim()).unwrap_or_else(|e| panic!("hex decode {}: {e}", entry.name))
}

#[test]
fn manifest_sha256_matches_decoded_fixtures() {
    let manifest_path = fixture_root().join("manifest.json");
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest"))
            .expect("manifest json");
    for entry in &manifest.fixtures {
        let bytes = load_fixture_bytes(entry);
        let digest = hex::encode(Sha256::digest(&bytes));
        assert_eq!(digest, entry.sha256, "SHA-256 mismatch for {}", entry.name);
    }
}

#[test]
fn golden_valid_frames_match_parser_expectations() {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(fixture_root().join("manifest.json")).unwrap(),
    )
    .unwrap();

    let by_name = |name: &str| {
        manifest
            .fixtures
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("missing fixture {name}"))
    };

    let empty = load_fixture_bytes(by_name("empty_8_byte"));
    let (tick, floats) =
        parse_readout_packet(&empty, DEFAULT_MAX_READOUT_FLOATS).expect("empty frame");
    assert_eq!(tick, 7);
    assert!(floats.is_empty());

    let one = load_fixture_bytes(by_name("one_float_12_byte"));
    let (tick, floats) = parse_readout_packet(&one, DEFAULT_MAX_READOUT_FLOATS).unwrap();
    assert_eq!(tick, 11);
    assert_eq!(floats, vec![1.5]);

    let sixteen = load_fixture_bytes(by_name("sixteen_float_72_byte"));
    let (_, floats) = parse_readout_packet(&sixteen, DEFAULT_MAX_READOUT_FLOATS).unwrap();
    assert_eq!(floats.len(), 16);

    let twenty = load_fixture_bytes(by_name("twenty_float_88_byte_historical_ambiguity"));
    let (tick, floats) = parse_readout_packet(&twenty, DEFAULT_MAX_READOUT_FLOATS).unwrap();
    assert_eq!(tick, 20);
    assert_eq!(
        floats,
        vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
            4.0, 4.25, 4.5, 4.75,
        ]
    );

    let at_max = load_fixture_bytes(by_name("readout_at_max_floats"));
    let (_, floats) = parse_readout_packet(&at_max, DEFAULT_MAX_READOUT_FLOATS).unwrap();
    assert_eq!(floats.len(), DEFAULT_MAX_READOUT_FLOATS);

    let over = load_fixture_bytes(by_name("readout_max_plus_one_float"));
    let err = parse_readout_packet(&over, DEFAULT_MAX_READOUT_FLOATS).expect_err("over limit");
    assert!(matches!(err, BackendError::InvalidInput(_)));
}

#[test]
fn golden_malformed_frames_rejected_without_backend_mutation() {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(fixture_root().join("manifest.json")).unwrap(),
    )
    .unwrap();

    for name in ["truncated_5_byte", "misaligned_9_byte"] {
        let entry = manifest.fixtures.iter().find(|f| f.name == name).unwrap();
        let bytes = load_fixture_bytes(entry);
        let err = parse_readout_packet(&bytes, DEFAULT_MAX_READOUT_FLOATS).expect_err(name);
        assert!(
            matches!(err, BackendError::CommunicationError(_)),
            "{name}: {err:?}"
        );

        let mut backend = ZmqIpcBackend::new();
        let seed = load_fixture_bytes(
            manifest
                .fixtures
                .iter()
                .find(|f| f.name == "one_float_12_byte")
                .unwrap(),
        );
        backend
            .apply_readout_packet_for_tests(&seed)
            .expect("seed cache");
        let before = backend.readout_cache_snapshot_for_tests();
        let apply_err = backend
            .apply_readout_packet_for_tests(&bytes)
            .expect_err(name);
        assert!(matches!(apply_err, BackendError::CommunicationError(_)));
        assert_eq!(backend.readout_cache_snapshot_for_tests(), before);
    }
}
