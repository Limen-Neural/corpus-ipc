// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn neuromodulator_rejects_non_finite() {
    for value in non_finite_values() {
        let mut snap = NeuromodulatorSnapshot {
            tick: 0,
            dopamine: 0.1,
            cortisol: 0.1,
            acetylcholine: 0.1,
            tempo: 1.0,
        };
        snap.dopamine = value;
        assert_eq!(snap.validate().unwrap_err().kind, ValidationKind::NonFinite);
    }
}

#[test]
fn stimulus_rejects_non_finite() {
    for value in non_finite_values() {
        let stimulus = StimulusBatch {
            values: vec![value],
            ..Default::default()
        };
        assert_eq!(
            stimulus.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
    }
}

#[test]
fn spike_event_rejects_non_finite() {
    for value in non_finite_values() {
        let spike = SpikeEvent {
            channel: 0,
            time: 0,
            strength: value,
        };
        assert_eq!(
            spike.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
    }
}

#[test]
fn embedding_rejects_non_finite() {
    for value in non_finite_values() {
        let embedding = EmbeddingBatch {
            embedding: vec![value],
            sequence_length: 1,
            ..Default::default()
        };
        assert_eq!(
            embedding.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
    }
}

#[test]
fn gradient_rejects_non_finite_values_and_traces() {
    for value in non_finite_values() {
        let mut update = sample_gradient_update();
        update.gradients = vec![value];
        assert_eq!(
            update.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
        update.gradients = vec![0.0];
        update.eligibility_trace = Some(vec![value]);
        assert_eq!(
            update.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
    }
}

#[test]
fn trace_rejects_non_finite() {
    for value in non_finite_values() {
        let trace = TraceData {
            channel_id: 1,
            trace_value: value,
            last_spike_time: 0,
        };
        assert_eq!(
            trace.validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
    }
}

#[test]
fn loss_and_config_reject_non_finite() {
    for value in non_finite_values() {
        assert_eq!(
            IpcMessage::Loss(value).validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
        assert_eq!(
            ConfigValue::Float(value).validate().unwrap_err().kind,
            ValidationKind::NonFinite
        );
        assert_eq!(
            ConfigValue::FloatArray(vec![value])
                .validate()
                .unwrap_err()
                .kind,
            ValidationKind::NonFinite
        );
    }
}
