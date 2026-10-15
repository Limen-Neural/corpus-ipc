// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::*;
use crate::{
    ConfigPayload, ConfigValue, EmbeddingBatch, GradientBatch, GradientUpdate, IpcMessage,
    NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch, TraceBatch, TraceData,
};

pub(super) fn unique_endpoint() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "ipc:///tmp/corpus_ipc_zmq_msg_{}_{}.ipc",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub(super) fn raw_publisher(endpoint: &str) -> zmq::Socket {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::PUB).expect("raw PUB socket");
    socket.set_linger(0).expect("linger");
    socket.bind(endpoint).expect("raw bind");
    socket
}

pub(super) fn raw_subscriber(endpoint: &str) -> zmq::Socket {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::SUB).expect("raw SUB socket");
    socket.set_linger(0).expect("linger");
    socket.set_subscribe(b"").expect("subscribe");
    socket.connect(endpoint).expect("raw connect");
    socket
}

pub(super) fn sample_messages() -> Vec<IpcMessage> {
    let mut config = HashMap::new();
    config.insert("gain".to_string(), ConfigValue::Float(0.5));
    vec![
        IpcMessage::Spikes(SpikeBatch {
            session_id: Some("session".to_string()),
            batch_id: 1,
            timestamp: 2,
            spikes: vec![SpikeEvent {
                channel: 3,
                time: 4,
                strength: 0.5,
            }],
            metadata: None,
        }),
        IpcMessage::Embeddings(EmbeddingBatch {
            session_id: None,
            batch_id: 7,
            embedding: vec![0.5],
            sequence_length: 1,
        }),
        IpcMessage::Stimuli(StimulusBatch {
            session_id: None,
            batch_id: 8,
            timestamp: 9,
            values: vec![0.25, 0.0],
            valid_mask: Some(vec![true, false]),
            metadata: None,
        }),
        IpcMessage::Neuromodulators(NeuromodulatorSnapshot {
            tick: 11,
            dopamine: 0.25,
            cortisol: 0.5,
            acetylcholine: 0.75,
            tempo: 1.0,
        }),
        IpcMessage::Loss(0.25),
        IpcMessage::ConfigUpdate(ConfigPayload {
            session_id: None,
            config,
        }),
        IpcMessage::GradientUpdate(sample_gradients()),
        IpcMessage::EligibilityTraces(sample_traces()),
        IpcMessage::TrainingComplete,
        IpcMessage::Shutdown,
        IpcMessage::Ping,
    ]
}

pub(super) fn sample_gradients() -> GradientBatch {
    GradientBatch {
        session_id: "session".to_string(),
        batch_id: 12,
        gradients: vec![GradientUpdate {
            layer_id: "layer".to_string(),
            gradients: vec![0.5],
            eligibility_trace: None,
        }],
    }
}

pub(super) fn sample_traces() -> TraceBatch {
    TraceBatch {
        session_id: "session".to_string(),
        batch_id: 13,
        traces: vec![TraceData {
            channel_id: 1,
            trace_value: 0.25,
            last_spike_time: 2,
        }],
    }
}

pub(super) fn spin_until(mut step: impl FnMut() -> bool, label: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if step() {
            return;
        }
    }
    panic!("timed out waiting for {label}");
}

pub(super) fn pace(last_send: &mut Instant, send: impl FnOnce()) {
    if last_send.elapsed() < Duration::from_millis(20) {
        return;
    }
    send();
    *last_send = Instant::now();
}

pub(super) fn idle_message(err: &ZmqMessageError) -> bool {
    matches!(err, ZmqMessageError::WouldBlock)
}

pub(super) fn progress<T, E: std::fmt::Display>(
    result: Result<T, E>,
    is_idle: impl Fn(&E) -> bool,
) -> Result<Option<T>, E> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) if is_idle(&err) => {
            thread::sleep(Duration::from_millis(1));
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

pub(super) fn expect_message(
    publisher: &ZmqMessagePublisher,
    subscriber: &ZmqMessageSubscriber,
    message: &IpcMessage,
) {
    let mut last_send = Instant::now() - Duration::from_secs(1);
    spin_until(
        || {
            pace(&mut last_send, || {
                publisher
                    .send(message)
                    .unwrap_or_else(|err| panic!("send failed: {err}"));
            });
            progress(subscriber.try_recv(), idle_message)
                .unwrap_or_else(|err| panic!("{err}"))
                .is_some_and(|got| &got == message)
        },
        "the published message",
    );
}

pub(super) fn recv_until_error(
    publisher: &zmq::Socket,
    subscriber: &ZmqMessageSubscriber,
    publish: impl Fn(&zmq::Socket),
) -> ZmqMessageError {
    let found = Cell::new(None);
    spin_until(
        || {
            publish(publisher);
            match progress(subscriber.try_recv(), idle_message) {
                Ok(None) => false,
                Ok(Some(message)) => panic!("malformed frame decoded as {message:?}"),
                Err(err) => {
                    found.set(Some(err));
                    true
                }
            }
        },
        "a typed receive error",
    );
    found.into_inner().expect("typed receive error")
}

#[test]
fn every_ipc_message_variant_round_trips_on_a_live_socket() {
    let endpoint = unique_endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let samples = sample_messages();
    assert_eq!(
        samples.len(),
        11,
        "one sample per current IpcMessage variant"
    );
    for message in &samples {
        expect_message(&publisher, &subscriber, message);
    }
}

#[test]
fn socket_options_are_visible_after_bind_and_connect() {
    let endpoint = unique_endpoint();
    let mut config = ZmqTransportConfig::new(&endpoint);
    config.send_hwm = 7;
    config.receive_hwm = 9;
    config.linger_ms = 25;
    config.reconnect_ivl_ms = 40;
    config.reconnect_ivl_max_ms = 80;

    let publisher = ZmqMessagePublisher::bind(config.clone()).unwrap();
    assert_eq!(
        publisher.socket_options_for_tests().unwrap(),
        (7, 9, 25, 40, 80)
    );
    let subscriber = ZmqMessageSubscriber::connect(config).unwrap();
    assert_eq!(
        subscriber.socket_options_for_tests().unwrap(),
        (7, 9, 25, 40, 80)
    );
}

#[test]
fn invalid_config_is_rejected_before_bind() {
    let cases = [
        (ZmqTransportConfig::new(""), "endpoint must be non-empty"),
        (
            ZmqTransportConfig::new("ipc://bad\0endpoint"),
            "endpoint must not contain an interior null",
        ),
    ];
    for (config, expected) in cases {
        let err = ZmqMessagePublisher::bind(config).unwrap_err();
        assert!(
            matches!(err, ZmqMessageError::InvalidConfig(message) if message == expected),
            "{err}"
        );
    }

    let mut negative_hwm = ZmqTransportConfig::new(unique_endpoint());
    negative_hwm.send_hwm = -1;
    assert!(matches!(
        ZmqMessagePublisher::bind(negative_hwm).unwrap_err(),
        ZmqMessageError::InvalidConfig("send_hwm must be >= 0")
    ));

    let mut negative_linger = ZmqTransportConfig::new(unique_endpoint());
    negative_linger.linger_ms = -2;
    assert!(matches!(
        ZmqMessageSubscriber::connect(negative_linger).unwrap_err(),
        ZmqMessageError::InvalidConfig("linger_ms must be >= -1")
    ));

    let mut zero_reconnect = ZmqTransportConfig::new(unique_endpoint());
    zero_reconnect.reconnect_ivl_ms = 0;
    assert!(matches!(
        ZmqMessagePublisher::bind(zero_reconnect).unwrap_err(),
        ZmqMessageError::InvalidConfig("reconnect_ivl_ms must be > 0")
    ));
}
