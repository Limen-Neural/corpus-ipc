// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::validation::{ProtocolLimits, Validate, ValidationKind, ValidationMeasure};
use serde::de::DeserializeOwned;

fn limits() -> ProtocolLimits {
    ProtocolLimits::DEFAULT
}

fn oversize_string(limit: usize) -> String {
    "a".repeat(limit.saturating_add(1))
}

fn max_string(limit: usize) -> String {
    "a".repeat(limit)
}

fn serde_err<T: DeserializeOwned>(json: serde_json::Value) -> String {
    match serde_json::from_value::<T>(json) {
        Ok(_) => panic!("payload must be rejected"),
        Err(err) => err.to_string(),
    }
}

fn assert_serde_kind<T: DeserializeOwned>(json: serde_json::Value, kind: ValidationKind) {
    let msg = serde_err::<T>(json);
    assert!(
        msg.contains(kind.as_str()),
        "expected {} in serde error, got {msg}",
        kind.as_str()
    );
}

fn sample_gradient_update() -> GradientUpdate {
    GradientUpdate {
        layer_id: "layer-a".into(),
        gradients: vec![0.1, 0.2],
        eligibility_trace: None,
    }
}

#[test]
fn every_limit_accepts_exact_max_and_rejects_max_plus_one() {
    let limits = limits();

    let mut stimulus = StimulusBatch {
        values: vec![0.0; limits.max_channel_values],
        ..Default::default()
    };
    stimulus.validate_with(limits).expect("channel max");
    stimulus.values.push(0.0);
    let err = stimulus.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, "values");
    assert_eq!(
        err.actual,
        Some(ValidationMeasure::Count(
            u64::try_from(limits.max_channel_values + 1).unwrap()
        ))
    );
    assert_eq!(
        err.limit,
        Some(ValidationMeasure::Count(
            u64::try_from(limits.max_channel_values).unwrap()
        ))
    );

    let mut spikes = SpikeBatch {
        spikes: vec![
            SpikeEvent {
                channel: 0,
                time: 0,
                strength: 1.0
            };
            limits.max_spike_events
        ],
        ..Default::default()
    };
    spikes.validate_with(limits).expect("spike max");
    spikes.spikes.push(SpikeEvent {
        channel: 1,
        time: 1,
        strength: 1.0,
    });
    let err = spikes.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, "spikes");

    let mut traces = TraceBatch {
        session_id: "s".into(),
        batch_id: 1,
        traces: (0..limits.max_traces)
            .map(|i| TraceData {
                channel_id: u16::try_from(i).unwrap_or(u16::MAX),
                trace_value: 0.1,
                last_spike_time: 0,
            })
            .collect(),
    };
    // max_traces is 65536, but channel_id is u16 so unique ids only exist for 65536 values
    // (0..=65535). Exact max is therefore also the full u16 space.
    traces.validate_with(limits).expect("trace max");
    traces.traces.push(TraceData {
        channel_id: 0,
        trace_value: 0.2,
        last_spike_time: 1,
    });
    let err = traces.validate_with(limits).unwrap_err();
    // max+1 is either limit or duplicate depending on uniqueness of the extra row.
    assert!(
        err.kind == ValidationKind::LimitExceeded
            || err.kind == ValidationKind::DuplicateIdentifier,
        "max+1 traces must fail: {err}"
    );

    let mut gradients = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: (0..limits.max_gradients)
            .map(|i| GradientUpdate {
                layer_id: format!("l{i}"),
                gradients: vec![0.0],
                eligibility_trace: None,
            })
            .collect(),
    };
    // max_gradients * 1 inner value plus row count exceeds max_aggregate_records (131072)
    // when max_gradients is 65536: 65536 rows + 65536 values = 131072, which is exact max.
    gradients.validate_with(limits).expect("gradient max");
    gradients.gradients.push(GradientUpdate {
        layer_id: "extra".into(),
        gradients: vec![0.0],
        eligibility_trace: None,
    });
    let err = gradients.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);

    let mut metadata = BatchMetadata::default();
    for i in 0..limits.max_metadata_entries {
        metadata.custom.insert(format!("k{i}"), "v".into());
    }
    metadata.validate_with(limits).expect("metadata max");
    metadata.custom.insert("overflow".into(), "v".into());
    let err = metadata.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, "custom");

    let session = SpikeBatch {
        session_id: Some(max_string(limits.max_string_bytes)),
        ..Default::default()
    };
    session.validate_with(limits).expect("string max");
    let mut oversize = session.clone();
    oversize.session_id = Some(oversize_string(limits.max_string_bytes));
    let err = oversize.validate_with(limits).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, "session_id");
    assert_eq!(
        err.actual,
        Some(ValidationMeasure::Bytes(
            u64::try_from(limits.max_string_bytes + 1).unwrap()
        ))
    );

    let mut aggregate = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: vec![GradientUpdate {
            layer_id: "a".into(),
            gradients: vec![0.0; 4],
            eligibility_trace: Some(vec![0.0; 4]),
        }],
    };
    let tight = ProtocolLimits {
        max_aggregate_records: 9,
        ..limits
    };
    aggregate.validate_with(tight).expect("aggregate max");
    aggregate.gradients[0].gradients.push(0.0);
    let err = aggregate.validate_with(tight).unwrap_err();
    assert_eq!(err.kind, ValidationKind::LimitExceeded);
    assert_eq!(err.path, "aggregate");
}

#[test]
fn nan_and_inf_rejected_in_each_float_payload() {
    let limits = limits();
    for (label, value) in [
        ("nan", f32::NAN),
        ("inf", f32::INFINITY),
        ("neg_inf", f32::NEG_INFINITY),
    ] {
        let mut snap = NeuromodulatorSnapshot {
            tick: 0,
            dopamine: 0.1,
            cortisol: 0.1,
            acetylcholine: 0.1,
            tempo: 1.0,
        };
        snap.dopamine = value;
        assert_eq!(
            snap.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} dopamine"
        );

        let stimulus = StimulusBatch {
            values: vec![value],
            ..Default::default()
        };
        assert_eq!(
            stimulus.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} stimulus"
        );

        let spike = SpikeEvent {
            channel: 0,
            time: 0,
            strength: value,
        };
        assert_eq!(
            spike.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} spike"
        );

        let embedding = EmbeddingBatch {
            embedding: vec![value],
            sequence_length: 1,
            ..Default::default()
        };
        assert_eq!(
            embedding.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} embedding"
        );

        let mut update = sample_gradient_update();
        update.gradients = vec![value];
        assert_eq!(
            update.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} gradient"
        );
        update.gradients = vec![0.0];
        update.eligibility_trace = Some(vec![value]);
        assert_eq!(
            update.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} eligibility"
        );

        let trace = TraceData {
            channel_id: 1,
            trace_value: value,
            last_spike_time: 0,
        };
        assert_eq!(
            trace.validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} trace"
        );

        assert_eq!(
            IpcMessage::Loss(value).validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} loss"
        );

        assert_eq!(
            ConfigValue::Float(value).validate().unwrap_err().kind,
            ValidationKind::NonFinite,
            "{label} config float"
        );
        assert_eq!(
            ConfigValue::FloatArray(vec![value])
                .validate()
                .unwrap_err()
                .kind,
            ValidationKind::NonFinite,
            "{label} config array"
        );
        let _ = limits;
    }
}

#[test]
fn deserialize_rejects_json_inf_in_float_payloads() {
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
    assert_serde_kind::<SpikeEvent>(
        serde_json::json!({"channel": 0, "time": 0, "strength": inf}),
        ValidationKind::NonFinite,
    );
    assert_serde_kind::<EmbeddingBatch>(
        serde_json::json!({
            "session_id": null,
            "batch_id": 0,
            "embedding": [inf],
            "sequence_length": 1
        }),
        ValidationKind::NonFinite,
    );
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
    assert_serde_kind::<IpcMessage>(serde_json::json!({"Loss": inf}), ValidationKind::NonFinite);
}

#[test]
fn mask_value_length_mismatch_is_typed() {
    let batch = StimulusBatch {
        values: vec![0.0, 1.0],
        valid_mask: Some(vec![true]),
        ..Default::default()
    };
    let err = batch.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::LengthMismatch);
    assert_eq!(err.actual, Some(ValidationMeasure::Count(1)));
    assert_eq!(err.limit, Some(ValidationMeasure::Count(2)));
}

#[test]
fn oversize_session_metadata_events_traces_gradients_rejected() {
    let limits = limits();
    let too_long = oversize_string(limits.max_string_bytes);

    let err = SpikeBatch {
        session_id: Some(too_long.clone()),
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "session_id");
    assert_eq!(err.kind, ValidationKind::LimitExceeded);

    let err = TraceBatch {
        session_id: too_long.clone(),
        batch_id: 1,
        traces: vec![],
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "session_id");

    let mut metadata = BatchMetadata::default();
    metadata.custom.insert(too_long.clone(), "v".into());
    assert_eq!(
        metadata.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
    metadata.custom.clear();
    metadata.custom.insert("k".into(), too_long.clone());
    assert_eq!(
        metadata.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
    metadata.custom.clear();
    metadata.source = Some(too_long.clone());
    assert_eq!(metadata.validate().unwrap_err().path, "source");

    let err = GradientUpdate {
        layer_id: too_long,
        gradients: vec![0.0],
        eligibility_trace: None,
    }
    .validate()
    .unwrap_err();
    assert_eq!(err.path, "layer_id");
}

#[test]
fn nested_metadata_and_duplicate_identifiers() {
    let mut nested = BatchMetadata::default();
    nested.custom.insert(String::new(), "v".into());
    let err = nested.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::NestedMetadata);

    let parent = StimulusBatch {
        metadata: Some({
            let mut meta = BatchMetadata::default();
            meta.custom.insert(String::new(), "v".into());
            meta
        }),
        ..Default::default()
    };
    assert_eq!(
        parent.validate().unwrap_err().kind,
        ValidationKind::NestedMetadata
    );

    let nested_object = serde_json::json!({
        "processing_latency_ns": null,
        "source": null,
        "custom": {"k": {"nested": true}}
    });
    assert!(
        serde_json::from_value::<BatchMetadata>(nested_object).is_err(),
        "nested object in metadata.custom must fail to deserialize"
    );

    let traces = TraceBatch {
        session_id: "s".into(),
        batch_id: 1,
        traces: vec![
            TraceData {
                channel_id: 3,
                trace_value: 0.1,
                last_spike_time: 0,
            },
            TraceData {
                channel_id: 3,
                trace_value: 0.2,
                last_spike_time: 1,
            },
        ],
    };
    let err = traces.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::DuplicateIdentifier);

    let gradients = GradientBatch {
        session_id: "s".into(),
        batch_id: 1,
        gradients: vec![
            GradientUpdate {
                layer_id: "shared".into(),
                gradients: vec![0.0],
                eligibility_trace: None,
            },
            GradientUpdate {
                layer_id: "shared".into(),
                gradients: vec![1.0],
                eligibility_trace: None,
            },
        ],
    };
    let err = gradients.validate().unwrap_err();
    assert_eq!(err.kind, ValidationKind::DuplicateIdentifier);

    assert_serde_kind::<TraceBatch>(
        serde_json::json!({
            "session_id": "s",
            "batch_id": 1,
            "traces": [
                {"channel_id": 1, "trace_value": 0.1, "last_spike_time": 0},
                {"channel_id": 1, "trace_value": 0.2, "last_spike_time": 1}
            ]
        }),
        ValidationKind::DuplicateIdentifier,
    );
}

#[test]
fn serde_rejects_over_limit_without_full_prose_match() {
    let limits = limits();
    let too_long = oversize_string(limits.max_string_bytes);
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

    let extra_value = vec![0.0; limits.max_channel_values + 1];
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
fn ipc_message_validate_dispatches_to_payload() {
    IpcMessage::Ping.validate().unwrap();
    IpcMessage::Shutdown.validate().unwrap();
    IpcMessage::TrainingComplete.validate().unwrap();
    IpcMessage::Loss(0.5).validate().unwrap();

    let mut batch = StimulusBatch {
        values: vec![0.0],
        valid_mask: Some(vec![true, false]),
        ..Default::default()
    };
    assert!(IpcMessage::Stimuli(batch.clone()).validate().is_err());
    batch.valid_mask = Some(vec![true]);
    IpcMessage::Stimuli(batch).validate().unwrap();
}

#[test]
fn same_checks_for_direct_construction_and_deserialization() {
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

#[test]
fn current_wire_fixtures_still_round_trip() {
    let spikes = IpcMessage::Spikes(SpikeBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        spikes: vec![SpikeEvent {
            channel: 3,
            time: 11,
            strength: 0.5,
        }],
        metadata: None,
    });
    let json = serde_json::to_value(&spikes).unwrap();
    let decoded: IpcMessage = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, spikes);
}

#[test]
fn duplicate_config_keys_and_oversize_config_map() {
    let raw = r#"{"session_id":null,"config":{"a":1.0,"a":2.0}}"#;
    let err = serde_json::from_str::<ConfigPayload>(raw)
        .expect_err("duplicate keys must be rejected")
        .to_string();
    assert!(
        err.contains(ValidationKind::DuplicateIdentifier.as_str()),
        "duplicate key error: {err}"
    );

    let mut payload = ConfigPayload {
        session_id: None,
        config: std::collections::HashMap::new(),
    };
    for i in 0..limits().max_metadata_entries {
        payload
            .config
            .insert(format!("k{i}"), ConfigValue::Boolean(true));
    }
    payload.validate().unwrap();
    payload
        .config
        .insert("overflow".into(), ConfigValue::Boolean(false));
    assert_eq!(
        payload.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
}
