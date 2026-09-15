// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn serde_rejects_inf_in_stimulus() {
    let inf = serde_json::json!(1e39);
    assert_serde_kind::<StimulusBatch>(
        serde_json::json!({
            "session_id": null,
            "batch_id": 0,
            "timestamp": 0,
            "values": [inf],
            "valid_mask": null,
            "metadata": null
        }),
        ValidationKind::NonFinite,
    );
}

#[test]
fn serde_rejects_inf_in_spike_event() {
    let inf = serde_json::json!(1e39);
    assert_serde_kind::<SpikeEvent>(
        serde_json::json!({"channel": 0, "time": 0, "strength": inf}),
        ValidationKind::NonFinite,
    );
}

#[test]
fn serde_rejects_inf_in_embedding() {
    let inf = serde_json::json!(1e39);
    assert_serde_kind::<EmbeddingBatch>(
        serde_json::json!({
            "session_id": null,
            "batch_id": 0,
            "embedding": [inf],
            "sequence_length": 1
        }),
        ValidationKind::NonFinite,
    );
}

#[test]
fn serde_rejects_inf_in_trace_and_gradient() {
    let inf = serde_json::json!(1e39);
    assert_serde_kind::<TraceData>(
        serde_json::json!({
            "channel_id": 1,
            "trace_value": inf,
            "last_spike_time": 0
        }),
        ValidationKind::NonFinite,
    );
    assert_serde_kind::<GradientUpdate>(
        serde_json::json!({
            "layer_id": "l0",
            "gradients": [inf]
        }),
        ValidationKind::NonFinite,
    );
}

#[test]
fn serde_rejects_inf_in_loss() {
    let inf = serde_json::json!(1e39);
    assert_serde_kind::<IpcMessage>(serde_json::json!({"Loss": inf}), ValidationKind::NonFinite);
}

#[test]
fn serde_rejects_oversize_session_id() {
    let too_long = oversize_string(limits().max_string_bytes);
    assert_serde_kind::<SpikeBatch>(
        serde_json::json!({
            "session_id": too_long,
            "batch_id": 0,
            "timestamp": 0,
            "spikes": [],
            "metadata": null
        }),
        ValidationKind::LimitExceeded,
    );
}

#[test]
fn serde_rejects_oversize_stimulus_values() {
    let extra_value = vec![0.0; limits().max_channel_values + 1];
    assert_serde_kind::<StimulusBatch>(
        serde_json::json!({
            "session_id": null,
            "batch_id": 0,
            "timestamp": 0,
            "values": extra_value,
            "valid_mask": null,
            "metadata": null
        }),
        ValidationKind::LimitExceeded,
    );
}

#[test]
fn serde_rejects_mismatched_mask() {
    assert_serde_kind::<StimulusBatch>(
        serde_json::json!({
            "session_id": null,
            "batch_id": 0,
            "timestamp": 0,
            "values": [0.0, 1.0],
            "valid_mask": [true],
            "metadata": null
        }),
        ValidationKind::LengthMismatch,
    );
}

#[test]
fn serde_and_direct_construction_share_range_checks() {
    let json = serde_json::json!({
        "tick": 1,
        "dopamine": 0.5,
        "cortisol": 0.5,
        "acetylcholine": 0.5,
        "tempo": 3.0
    });
    assert_serde_kind::<NeuromodulatorSnapshot>(json, ValidationKind::OutOfRange);

    let mut snap = NeuromodulatorSnapshot {
        tick: 1,
        dopamine: 0.5,
        cortisol: 0.5,
        acetylcholine: 0.5,
        tempo: 3.0,
    };
    assert_eq!(
        snap.validate().unwrap_err().kind,
        ValidationKind::OutOfRange
    );
    snap.tempo = 1.0;
    snap.validate().unwrap();
}
