// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn max_trace_batch(limits: ProtocolLimits) -> TraceBatch {
    TraceBatch {
        session_id: "s".into(),
        batch_id: 1,
        traces: (0..limits.max_traces)
            .map(|i| TraceData {
                channel_id: u16::try_from(i).unwrap_or(u16::MAX),
                trace_value: 0.1,
                last_spike_time: 0,
            })
            .collect(),
    }
}

#[test]
fn traces_accept_exact_max() {
    let limits = limits();
    // max_traces is 65536, and channel_id is u16, so exact max fills 0..=65535.
    max_trace_batch(limits)
        .validate_with(limits)
        .expect("trace max");
}

#[test]
fn traces_reject_max_plus_one() {
    let limits = limits();
    let mut traces = max_trace_batch(limits);
    traces.traces.push(TraceData {
        channel_id: 0,
        trace_value: 0.2,
        last_spike_time: 1,
    });
    let err = traces.validate_with(limits).unwrap_err();
    // Extra row is either over the count cap or a duplicate channel_id.
    assert!(
        err.kind == ValidationKind::LimitExceeded
            || err.kind == ValidationKind::DuplicateIdentifier,
        "max+1 traces must fail: {err}"
    );
}

#[test]
fn duplicate_channel_id_is_rejected() {
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
}

#[test]
fn serde_rejects_duplicate_channel_id() {
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
fn trace_batch_round_trips() {
    let traces = TraceBatch {
        session_id: "s".into(),
        batch_id: 1,
        traces: vec![TraceData {
            channel_id: 4,
            trace_value: 0.2,
            last_spike_time: 1,
        }],
    };
    let json = serde_json::to_value(&traces).unwrap();
    let decoded: TraceBatch = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.traces[0].channel_id, 4);
    IpcMessage::EligibilityTraces(decoded).validate().unwrap();
}
