// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::tests::{
    expect_message, idle_message, pace, progress, raw_publisher, recv_until_error, spin_until,
    unique_endpoint,
};
use super::*;
use crate::compatibility::CompatibilityError;
use crate::{EnvelopeError, IpcMessage, StimulusBatch};

#[test]
fn publisher_rejects_invalid_payloads_without_sending() {
    let endpoint = unique_endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    expect_message(&publisher, &subscriber, &IpcMessage::Ping);
    // `expect_message` resends until it sees the message, so duplicates may
    // still be in flight or queued. Drain them before asserting silence.
    let mut quiet = 0;
    while quiet < 3 {
        thread::sleep(Duration::from_millis(30));
        match subscriber.try_recv() {
            Ok(_) => quiet = 0,
            Err(ZmqMessageError::WouldBlock) => quiet += 1,
            Err(err) => panic!("{err}"),
        }
    }

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

fn assert_subscriber_drains_to_idle(subscriber: &ZmqMessageSubscriber) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match subscriber.try_recv() {
            Err(ZmqMessageError::WouldBlock) => return,
            Err(ZmqMessageError::UnexpectedFraming(FramingFault::Multipart)) => continue,
            other => panic!("subscriber did not go idle: {other:?}"),
        }
    }
    panic!("timed out draining multipart leftovers");
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
    // `recv_until_error` republishes, so another multipart may already be queued.
    assert_subscriber_drains_to_idle(&subscriber);

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

#[test]
fn oversized_multipart_closes_the_subscriber() {
    let endpoint = unique_endpoint();
    let publisher = raw_publisher(&endpoint);
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    let err = recv_until_error(&publisher, &subscriber, |socket| {
        let parts = super::transport::MULTIPART_DRAIN_LIMIT + 1;
        for _ in 0..parts {
            socket
                .send(&b"part"[..], zmq::SNDMORE | zmq::DONTWAIT)
                .unwrap();
        }
        socket.send(&b"tail"[..], zmq::DONTWAIT).unwrap();
    });
    assert!(matches!(
        err,
        ZmqMessageError::UnexpectedFraming(FramingFault::Multipart)
    ));
    assert!(matches!(
        subscriber.try_recv(),
        Err(ZmqMessageError::Closed)
    ));
}
