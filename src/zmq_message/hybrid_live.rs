// SPDX-License-Identifier: MIT OR Apache-2.0

use std::thread;
use std::time::{Duration, Instant};

use super::tests::{
    idle_message, pace, progress, raw_subscriber, sample_gradients, sample_messages, sample_traces,
    spin_until, unique_endpoint,
};
use super::*;
use crate::{
    EmbeddingBatch, GradientBatch, HybridFlowBackend, IpcMessage, ProtocolError, SpikeBatch,
    SpikeEvent, encode_canonical_ipc_message,
};

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
