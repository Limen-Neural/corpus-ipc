// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::*;
use crate::compatibility::CompatibilityError;
use crate::{
    ConfigPayload, ConfigValue, EmbeddingBatch, EnvelopeError, GradientBatch, GradientUpdate,
    HybridFlowBackend, IpcMessage, NeuromodulatorSnapshot, ProtocolError, SpikeBatch, SpikeEvent,
    StimulusBatch, TraceBatch, TraceData, encode_canonical_ipc_message,
};

fn unique_endpoint() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "ipc:///tmp/corpus_ipc_zmq_msg_{}_{}.ipc",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn raw_publisher(endpoint: &str) -> zmq::Socket {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::PUB).expect("raw PUB socket");
    socket.set_linger(0).expect("linger");
    socket.bind(endpoint).expect("raw bind");
    socket
}

fn raw_subscriber(endpoint: &str) -> zmq::Socket {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::SUB).expect("raw SUB socket");
    socket.set_linger(0).expect("linger");
    socket.set_subscribe(b"").expect("subscribe");
    socket.connect(endpoint).expect("raw connect");
    socket
}

fn sample_messages() -> Vec<IpcMessage> {
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

fn sample_gradients() -> GradientBatch {
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

fn sample_traces() -> TraceBatch {
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

fn spin_until(mut step: impl FnMut() -> bool, label: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if step() {
            return;
        }
    }
    panic!("timed out waiting for {label}");
}

fn pace(last_send: &mut Instant, send: impl FnOnce()) {
    if last_send.elapsed() < Duration::from_millis(20) {
        return;
    }
    send();
    *last_send = Instant::now();
}

fn idle_message(err: &ZmqMessageError) -> bool {
    matches!(err, ZmqMessageError::WouldBlock)
}

fn progress<T, E: std::fmt::Display>(
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

fn expect_message(
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

fn recv_until_error(
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

#[test]
fn publisher_rejects_invalid_payloads_without_sending() {
    let endpoint = unique_endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    expect_message(&publisher, &subscriber, &IpcMessage::Ping);

    let invalid = IpcMessage::Stimuli(StimulusBatch {
        session_id: None,
        batch_id: 1,
        timestamp: 0,
        values: vec![0.5],
        valid_mask: Some(vec![true, false]),
        metadata: None,
    });
    assert!(matches!(
        publisher.send(&invalid).unwrap_err(),
        ZmqMessageError::Validation(_)
    ));
    thread::sleep(Duration::from_millis(30));
    assert!(matches!(
        subscriber.try_recv(),
        Err(ZmqMessageError::WouldBlock)
    ));
}

#[test]
fn slow_joiner_loss_drops_frames_sent_before_subscribe() {
    let endpoint = unique_endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    publisher.send(&IpcMessage::Loss(1.0)).unwrap();

    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let mut last_send = Instant::now() - Duration::from_secs(1);
    spin_until(
        || {
            pace(&mut last_send, || {
                publisher.send(&IpcMessage::Loss(2.0)).unwrap();
            });
            match progress(subscriber.try_recv(), idle_message) {
                Ok(Some(IpcMessage::Loss(value))) => value == 2.0,
                Ok(Some(other)) => panic!("unexpected message {other:?}"),
                Ok(None) => false,
                Err(err) => panic!("{err}"),
            }
        },
        "the post-subscribe frame",
    );
}

#[test]
fn empty_receive_is_eagain() {
    let subscriber =
        ZmqMessageSubscriber::connect(ZmqTransportConfig::new(unique_endpoint())).unwrap();
    assert!(matches!(
        subscriber.try_recv(),
        Err(ZmqMessageError::WouldBlock)
    ));
}

#[test]
fn malformed_json_is_an_envelope_error() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket.send("{not-json", zmq::DONTWAIT).unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::Envelope(EnvelopeError::Json(_))
    ));
}

#[test]
fn incompatible_wire_version_fails_closed() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket
            .send(
                &br#"{"payload":"Ping","wire_version":2}"#[..],
                zmq::DONTWAIT,
            )
            .unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::Envelope(EnvelopeError::Compatibility(
            CompatibilityError::TooNew { .. }
        ))
    ));

    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket
            .send(
                &br#"{"payload":"Ping","wire_version":0}"#[..],
                zmq::DONTWAIT,
            )
            .unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::Envelope(EnvelopeError::Compatibility(
            CompatibilityError::TooOld { .. }
        ))
    ));
}

#[test]
fn invalid_payload_fails_validation_on_decode() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let frame = br#"{"payload":{"Neuromodulators":{"acetylcholine":0.0,"cortisol":0.0,"dopamine":5.0,"tempo":1.0,"tick":1}},"wire_version":1}"#;
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket.send(&frame[..], zmq::DONTWAIT).unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::Envelope(EnvelopeError::Payload(_))
    ));
}

#[test]
fn multipart_and_empty_frames_are_framing_errors() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket
            .send(
                &br#"{"payload":"Ping","wire_version":1}"#[..],
                zmq::SNDMORE | zmq::DONTWAIT,
            )
            .unwrap();
        socket
            .send(
                &br#"{"payload":"Shutdown","wire_version":1}"#[..],
                zmq::DONTWAIT,
            )
            .unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::UnexpectedFraming(FramingFault::Multipart)
    ));

    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket.send(&b""[..], zmq::DONTWAIT).unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::UnexpectedFraming(FramingFault::Empty)
    ));
}

#[test]
fn binary_readout_bytes_are_not_a_structured_message() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let mut frame = 1_i64.to_le_bytes().to_vec();
    frame.extend_from_slice(&1.0_f32.to_le_bytes());
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        socket.send(frame.as_slice(), zmq::DONTWAIT).unwrap();
    });
    assert!(matches!(err, ZmqMessageError::Envelope(_)));
}

#[test]
fn high_water_mark_pressure_drops_without_panic() {
    let endpoint = unique_endpoint();
    let mut outbound = ZmqTransportConfig::new(&endpoint);
    outbound.send_hwm = 1;
    outbound.receive_hwm = 1;
    outbound.linger_ms = 0;
    let inbound = outbound.clone();
    let publisher = ZmqMessagePublisher::bind(outbound).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(inbound).unwrap();
    expect_message(&publisher, &subscriber, &IpcMessage::Ping);

    let (accepted, blocked) = count_hwm_sends(&publisher);
    thread::sleep(Duration::from_millis(50));
    let delivered = count_deliveries(&subscriber);
    assert!(
        delivered < accepted || blocked > 0,
        "HWM pressure must drop or surface EAGAIN; accepted={accepted} delivered={delivered} blocked={blocked}"
    );
}

fn note_hwm_send(result: Result<(), ZmqMessageError>, accepted: &mut u32, blocked: &mut u32) {
    match result {
        Ok(()) => *accepted += 1,
        Err(ZmqMessageError::HighWaterMark) => *blocked += 1,
        Err(err) => panic!("{err}"),
    }
}

fn count_hwm_sends(publisher: &ZmqMessagePublisher) -> (u32, u32) {
    let mut accepted = 0_u32;
    let mut blocked = 0_u32;
    for index in 0..100 {
        note_hwm_send(
            publisher.send(&IpcMessage::Loss(index as f32)),
            &mut accepted,
            &mut blocked,
        );
    }
    (accepted, blocked)
}

fn count_deliveries(subscriber: &ZmqMessageSubscriber) -> u32 {
    let mut delivered = 0_u32;
    let drain_deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < drain_deadline {
        let arrived = progress(subscriber.try_recv(), idle_message)
            .unwrap_or_else(|err| panic!("{err}"))
            .is_some();
        delivered += u32::from(arrived);
    }
    delivered
}

#[test]
fn close_and_drop_do_not_panic() {
    let endpoint = unique_endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    publisher.close();
    subscriber.close();
    assert!(matches!(
        publisher.send(&IpcMessage::Ping),
        Err(ZmqMessageError::Closed)
    ));
    assert!(matches!(
        subscriber.try_recv(),
        Err(ZmqMessageError::Closed)
    ));
    publisher.close();
    subscriber.close();

    let started = Instant::now();
    drop(ZmqMessagePublisher::bind(ZmqTransportConfig::new(unique_endpoint())).unwrap());
    assert!(started.elapsed() < Duration::from_secs(1));

    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(unique_endpoint())).unwrap();
    thread::spawn(move || drop(publisher))
        .join()
        .expect("owner thread panicked");

    let publisher =
        Arc::new(ZmqMessagePublisher::bind(ZmqTransportConfig::new(unique_endpoint())).unwrap());
    let shared = Arc::clone(&publisher);
    thread::spawn(move || {
        shared.send(&IpcMessage::Ping).unwrap();
    })
    .join()
    .expect("shared publisher panicked");
}

struct HybridPair {
    backend: ZmqHybridFlowBackend,
    peer_out: ZmqMessageSubscriber,
    peer_in: ZmqMessagePublisher,
    outbound_endpoint: String,
}

fn hybrid_pair() -> HybridPair {
    let outbound_endpoint = unique_endpoint();
    let inbound = unique_endpoint();
    let peer_in = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&inbound)).unwrap();
    let backend = ZmqHybridFlowBackend::open(
        ZmqTransportConfig::new(&outbound_endpoint),
        ZmqTransportConfig::new(&inbound),
    )
    .unwrap();
    let peer_out =
        ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&outbound_endpoint)).unwrap();
    HybridPair {
        backend,
        peer_out,
        peer_in,
        outbound_endpoint,
    }
}

fn dispatch_outbound(pair: &mut HybridPair, message: &IpcMessage) {
    match message {
        IpcMessage::Spikes(batch) => pair.backend.send_spikes(batch.clone()).unwrap(),
        IpcMessage::Embeddings(batch) => pair.backend.send_embeddings(batch.clone()).unwrap(),
        other => panic!("peer expectation only covers outbound variants, got {other:?}"),
    }
}

#[test]
fn hybrid_flow_uses_canonical_frames_for_spikes_and_embeddings() {
    let mut pair = hybrid_pair();
    let spikes = match &sample_messages()[0] {
        IpcMessage::Spikes(batch) => batch.clone(),
        other => panic!("sample 0 is spikes, got {other:?}"),
    };
    let message = IpcMessage::Spikes(spikes.clone());
    let expected = encode_canonical_ipc_message(&message).unwrap();
    let raw = raw_subscriber(&pair.outbound_endpoint);
    let mut last_send = Instant::now() - Duration::from_secs(1);
    spin_until(
        || {
            pace(&mut last_send, || dispatch_outbound(&mut pair, &message));
            progress(pair.peer_out.try_recv(), idle_message)
                .unwrap_or_else(|err| panic!("{err}"))
                .is_some_and(|got| got.eq(&message))
        },
        "spike observation",
    );

    let embeddings = IpcMessage::Embeddings(EmbeddingBatch {
        session_id: None,
        batch_id: 4,
        embedding: vec![0.5],
        sequence_length: 1,
    });
    spin_until(
        || {
            pace(&mut last_send, || dispatch_outbound(&mut pair, &embeddings));
            progress(pair.peer_out.try_recv(), idle_message)
                .unwrap_or_else(|err| panic!("{err}"))
                .is_some_and(|got| got.eq(&embeddings))
        },
        "embedding observation",
    );

    spin_until(
        || {
            pair.backend.send_spikes(spikes.clone()).unwrap();
            progress(raw.recv_bytes(zmq::DONTWAIT), |err| {
                matches!(err, zmq::Error::EAGAIN)
            })
            .unwrap_or_else(|err| panic!("{err}"))
            .is_some_and(|frame| frame == expected)
        },
        "canonical encoder bytes",
    );
}

fn inbound_is_quiet(
    result: Result<Option<GradientBatch>, crate::BackendError>,
    quiet_since: &mut Instant,
) -> bool {
    match result {
        Ok(None) => quiet_since.elapsed() >= Duration::from_millis(40),
        Ok(Some(_)) | Err(crate::BackendError::Protocol(_)) => {
            *quiet_since = Instant::now();
            false
        }
        Err(err) => panic!("settle failed: {err}"),
    }
}

fn settle_inbound(pair: &mut HybridPair) {
    let started = Instant::now();
    let mut quiet_since = Instant::now();
    while started.elapsed() < Duration::from_millis(300) {
        if inbound_is_quiet(pair.backend.try_receive_gradients(), &mut quiet_since) {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn poll_peer<T>(
    pair: &mut HybridPair,
    message: &IpcMessage,
    mut accept: impl FnMut(&mut HybridPair) -> Option<T>,
    label: &str,
) -> T {
    let mut last_send = Instant::now() - Duration::from_secs(1);
    let mut found = None;
    spin_until(
        || {
            pace(&mut last_send, || pair.peer_in.send(message).unwrap());
            if let Some(value) = accept(pair) {
                found = Some(value);
                return true;
            }
            false
        },
        label,
    );
    found.expect(label)
}

fn poll_gradients(pair: &mut HybridPair) -> Result<Option<GradientBatch>, crate::BackendError> {
    let message = IpcMessage::GradientUpdate(sample_gradients());
    poll_peer(
        pair,
        &message,
        |pair| match pair.backend.try_receive_gradients() {
            Ok(None) => {
                thread::sleep(Duration::from_millis(1));
                None
            }
            other => Some(other),
        },
        "a gradient",
    )
}

fn wait_for_unexpected_ping(pair: &mut HybridPair) {
    poll_peer(
        pair,
        &IpcMessage::Ping,
        |pair| match pair.backend.try_receive_gradients() {
            Err(crate::BackendError::Protocol(ProtocolError::UnexpectedVariant {
                found: "Ping",
                expected: "GradientUpdate",
            })) => Some(()),
            Ok(None) => {
                thread::sleep(Duration::from_millis(1));
                None
            }
            Ok(Some(_)) => None,
            Err(err) => panic!("unexpected gradient outcome: {err:?}"),
        },
        "unexpected Ping",
    );
}

fn wait_for_traces(pair: &mut HybridPair) {
    let message = IpcMessage::EligibilityTraces(sample_traces());
    poll_peer(
        pair,
        &message,
        |pair| match pair.backend.try_receive_traces() {
            Ok(Some(batch)) => {
                assert_eq!(batch, sample_traces());
                Some(())
            }
            Ok(None) => {
                thread::sleep(Duration::from_millis(1));
                None
            }
            Err(crate::BackendError::Protocol(ProtocolError::UnexpectedVariant {
                found: "Ping" | "GradientUpdate",
                ..
            })) => None,
            Err(err) => panic!("{err}"),
        },
        "eligibility traces",
    );
}

#[test]
fn hybrid_flow_receives_gradients_and_traces_only() {
    let mut pair = hybrid_pair();
    assert!(matches!(pair.backend.try_receive_gradients(), Ok(None)));
    assert!(matches!(pair.backend.try_receive_traces(), Ok(None)));

    let received = poll_gradients(&mut pair).unwrap();
    assert_eq!(received, Some(sample_gradients()));
    settle_inbound(&mut pair);
    wait_for_unexpected_ping(&mut pair);
    settle_inbound(&mut pair);
    wait_for_traces(&mut pair);
}

#[test]
fn hybrid_flow_reports_invalid_payloads_and_closed_sockets() {
    let mut pair = hybrid_pair();
    let mut spikes = SpikeBatch {
        session_id: None,
        batch_id: 1,
        timestamp: 0,
        spikes: vec![SpikeEvent {
            channel: 1,
            time: 1,
            strength: 0.5,
        }],
        metadata: None,
    };
    spikes.spikes[0].strength = f32::NAN;
    assert!(matches!(
        pair.backend.send_spikes(spikes),
        Err(crate::BackendError::InvalidInput(_))
    ));

    pair.backend.close();
    assert!(matches!(
        pair.backend.send_embeddings(EmbeddingBatch {
            session_id: None,
            batch_id: 1,
            embedding: vec![0.5],
            sequence_length: 1,
        }),
        Err(crate::BackendError::CommunicationError(_))
    ));
    assert!(matches!(
        pair.backend.try_receive_gradients(),
        Err(crate::BackendError::CommunicationError(_))
    ));
}
