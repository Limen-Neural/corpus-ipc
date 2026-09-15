// SPDX-License-Identifier: MIT OR Apache-2.0

use super::neuromod::NeuromodulatorSnapshotWire;
use super::*;
use crate::validation::Validate;

fn sample_spike_batch() -> SpikeBatch {
    SpikeBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        spikes: vec![SpikeEvent {
            channel: 3,
            time: 11,
            strength: 0.5,
        }],
        metadata: None,
    }
}

fn sample_stimulus_batch() -> StimulusBatch {
    StimulusBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        values: vec![0.5, 0.0, -0.25],
        valid_mask: Some(vec![true, false, true]),
        metadata: None,
    }
}

fn sample_neuromodulator_snapshot() -> NeuromodulatorSnapshot {
    NeuromodulatorSnapshot {
        tick: 42,
        dopamine: 0.4,
        cortisol: 0.3,
        acetylcholine: 0.2,
        tempo: 1.0,
    }
}

#[test]
fn neuromodulator_snapshot_validate_accepts_in_range_values() {
    assert!(sample_neuromodulator_snapshot().validate().is_ok());
}

#[test]
fn neuromodulator_snapshot_from_scores_accepts_in_range_values() {
    let snap = NeuromodulatorSnapshot::from_scores(1, &[0.4, 0.3, 0.2, 1.0])
        .expect("in-range scores must construct");
    assert_eq!(snap.tick, 1);
    assert!((snap.dopamine - 0.4).abs() < 1e-6);
}

#[test]
fn neuromodulator_snapshot_from_scores_rejects_out_of_range_value() {
    // tempo (scores[3]) outside documented [0.5, 2.0]
    let err = NeuromodulatorSnapshot::from_scores(1, &[0.4, 0.3, 0.2, 3.0])
        .expect_err("out-of-range tempo byte must fail construction, not just deserialization");
    assert_eq!(err.path, "tempo");
    assert_eq!(err.kind, crate::validation::ValidationKind::OutOfRange);
}

#[test]
fn neuromodulator_snapshot_validate_rejects_out_of_range_value() {
    let mut snap = sample_neuromodulator_snapshot();
    snap.tempo = 3.0; // outside documented [0.5, 2.0]
    let err = snap
        .validate()
        .expect_err("out-of-range tempo must fail validation");
    assert_eq!(err.path, "tempo");
    assert_eq!(err.kind, crate::validation::ValidationKind::OutOfRange);
}

#[test]
fn neuromodulator_snapshot_validate_rejects_non_finite_value() {
    let mut snap = sample_neuromodulator_snapshot();
    snap.dopamine = f32::NAN;
    let err = snap
        .validate()
        .expect_err("non-finite dopamine must fail validation");
    assert_eq!(err.path, "dopamine");
    assert_eq!(err.kind, crate::validation::ValidationKind::NonFinite);
}

#[test]
fn neuromodulator_snapshot_deserialize_rejects_out_of_range_value() {
    let json = serde_json::json!({
        "tick": 1,
        "dopamine": 0.5,
        "cortisol": 0.5,
        "acetylcholine": 0.5,
        "tempo": 3.0
    });
    let result: Result<NeuromodulatorSnapshot, _> = serde_json::from_value(json);
    assert!(
        result.is_err(),
        "an out-of-range tempo must fail to deserialize"
    );
}

#[test]
fn neuromodulator_snapshot_try_from_wire_rejects_non_finite_value() {
    // JSON has no NaN/Infinity literal, so this exercises the TryFrom
    // conversion that backs Deserialize directly, for wire formats
    // (e.g. bincode) that can represent a non-finite f32.
    let wire = NeuromodulatorSnapshotWire {
        tick: 1,
        dopamine: f32::NAN,
        cortisol: 0.5,
        acetylcholine: 0.5,
        tempo: 1.0,
    };
    assert!(
        NeuromodulatorSnapshot::try_from(wire).is_err(),
        "a non-finite dopamine must fail the TryFrom conversion"
    );
}

fn sample_trace_batch() -> TraceBatch {
    TraceBatch {
        session_id: "sess-1".into(),
        batch_id: 7,
        traces: vec![TraceData {
            channel_id: 3,
            trace_value: 0.25,
            last_spike_time: 11,
        }],
    }
}

#[test]
fn ipc_aliases_are_the_same_types() {
    fn as_ipc_spike(batch: IpcSpikeBatch) -> SpikeBatch {
        batch
    }
    fn as_ipc_trace(batch: IpcTraceBatch) -> TraceBatch {
        batch
    }
    let _ = as_ipc_spike(sample_spike_batch());
    let _ = as_ipc_trace(sample_trace_batch());
}

#[test]
fn spike_batch_json_keys_stay_stable() {
    let json = serde_json::to_value(sample_spike_batch()).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "session_id": "sess-1",
            "batch_id": 7,
            "timestamp": 1_700_000_000,
            "spikes": [{
                "channel": 3,
                "time": 11,
                "strength": 0.5
            }],
            "metadata": null
        })
    );
}

#[test]
fn trace_batch_json_keys_stay_stable() {
    let json = serde_json::to_value(sample_trace_batch()).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "session_id": "sess-1",
            "batch_id": 7,
            "traces": [{
                "channel_id": 3,
                "trace_value": 0.25,
                "last_spike_time": 11
            }]
        })
    );
}

#[test]
fn ipc_message_envelopes_keep_variant_names() {
    let spikes = serde_json::to_value(IpcMessage::Spikes(sample_spike_batch())).unwrap();
    let traces = serde_json::to_value(IpcMessage::EligibilityTraces(sample_trace_batch())).unwrap();
    assert!(spikes.get("Spikes").is_some());
    assert!(traces.get("EligibilityTraces").is_some());
    let decoded_spikes: IpcMessage = serde_json::from_value(spikes).unwrap();
    let decoded_traces: IpcMessage = serde_json::from_value(traces).unwrap();
    assert_eq!(decoded_spikes, IpcMessage::Spikes(sample_spike_batch()));
    assert_eq!(
        decoded_traces,
        IpcMessage::EligibilityTraces(sample_trace_batch())
    );
}

#[test]
fn stimulus_batch_json_keys_stay_stable() {
    let json = serde_json::to_value(sample_stimulus_batch()).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "session_id": "sess-1",
            "batch_id": 7,
            "timestamp": 1_700_000_000,
            "values": [0.5, 0.0, -0.25],
            "valid_mask": [true, false, true],
            "metadata": null
        })
    );
}

#[test]
fn stimulus_batch_round_trips() {
    let batch = sample_stimulus_batch();
    let json = serde_json::to_value(&batch).unwrap();
    let decoded: StimulusBatch = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, batch);
}

#[test]
fn stimulus_batch_default_has_no_mask_and_all_channels_valid() {
    let batch = StimulusBatch::default();
    assert!(batch.values.is_empty());
    assert!(batch.valid_mask.is_none());
}

#[test]
fn stimulus_batch_validate_accepts_matching_or_absent_mask() {
    assert!(StimulusBatch::default().validate().is_ok());
    assert!(sample_stimulus_batch().validate().is_ok());
}

#[test]
fn stimulus_batch_validate_rejects_mismatched_mask_length() {
    let batch = StimulusBatch {
        values: vec![0.0, 0.0, 0.0],
        valid_mask: Some(vec![true, false]),
        ..Default::default()
    };
    let err = batch
        .validate()
        .expect_err("mismatched mask must fail validation");
    assert_eq!(err.path, "valid_mask");
    assert_eq!(err.kind, crate::validation::ValidationKind::LengthMismatch);
    assert_eq!(
        err.actual,
        Some(crate::validation::ValidationMeasure::Count(2))
    );
    assert_eq!(
        err.limit,
        Some(crate::validation::ValidationMeasure::Count(3))
    );
}

#[test]
fn stimulus_batch_deserialize_rejects_mismatched_mask_length() {
    let json = serde_json::json!({
        "session_id": null,
        "batch_id": 1,
        "timestamp": 0,
        "values": [0.0, 1.0],
        "valid_mask": [true],
        "metadata": null
    });
    let result: Result<StimulusBatch, _> = serde_json::from_value(json);
    assert!(
        result.is_err(),
        "a valid_mask shorter than values must fail to deserialize"
    );
}

#[test]
fn stimulus_batch_invalid_channel_is_distinct_from_a_real_zero() {
    // Channel 1 is a genuine zero reading; channel 2 is missing/invalid and
    // its 0.0 placeholder must not be mistaken for a real reading.
    let batch = StimulusBatch {
        session_id: None,
        batch_id: 1,
        timestamp: 0,
        values: vec![1.0, 0.0, 0.0],
        valid_mask: Some(vec![true, true, false]),
        metadata: None,
    };
    let json = serde_json::to_value(&batch).unwrap();
    let decoded: StimulusBatch = serde_json::from_value(json).unwrap();
    let mask = decoded.valid_mask.expect("mask must survive round-trip");
    assert!(mask[1], "channel 1 is a valid, genuine zero reading");
    assert!(!mask[2], "channel 2 is invalid/missing, not a real zero");
}

#[test]
fn ipc_message_stimuli_and_neuromodulators_keep_variant_names() {
    let stimuli = serde_json::to_value(IpcMessage::Stimuli(sample_stimulus_batch())).unwrap();
    let neuromods =
        serde_json::to_value(IpcMessage::Neuromodulators(sample_neuromodulator_snapshot()))
            .unwrap();
    assert!(stimuli.get("Stimuli").is_some());
    assert!(neuromods.get("Neuromodulators").is_some());
    let decoded_stimuli: IpcMessage = serde_json::from_value(stimuli).unwrap();
    let decoded_neuromods: IpcMessage = serde_json::from_value(neuromods).unwrap();
    assert_eq!(
        decoded_stimuli,
        IpcMessage::Stimuli(sample_stimulus_batch())
    );
    assert_eq!(
        decoded_neuromods,
        IpcMessage::Neuromodulators(sample_neuromodulator_snapshot())
    );
}
