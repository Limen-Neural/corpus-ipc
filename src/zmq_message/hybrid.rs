// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    BackendError, EmbeddingBatch, GradientBatch, HybridFlowBackend, IpcMessage, ProtocolError,
    SpikeBatch, TraceBatch,
};

use super::transport::{
    self, ZmqMessageError, ZmqMessagePublisher, ZmqMessageSubscriber, ZmqTransportConfig,
};

/// [`HybridFlowBackend`] adapter over a structured publisher/subscriber pair.
///
/// Outbound spikes and embeddings are [`IpcMessage::Spikes`] and
/// [`IpcMessage::Embeddings`] sent through [`ZmqMessagePublisher`]. Inbound
/// gradients and eligibility traces are [`IpcMessage::GradientUpdate`] and
/// [`IpcMessage::EligibilityTraces`] received through [`ZmqMessageSubscriber`].
/// Framing is not reimplemented here.
///
/// Any other inbound variant returns [`BackendError::Protocol`]
/// ([`ProtocolError::UnexpectedVariant`]). The frame is consumed; it is not
/// discarded quietly and it is not pushed back onto the socket.
///
/// This type does not implement [`crate::IpcBackend`]. Binary readouts stay on
/// [`crate::ZmqIpcBackend`].
///
/// Delivery is the publisher/subscriber contract: best-effort, at-most-once,
/// no acknowledgements. See the [module docs](crate::zmq_message).
pub struct ZmqHybridFlowBackend {
    outbound: ZmqMessagePublisher,
    inbound: ZmqMessageSubscriber,
}

impl std::fmt::Debug for ZmqHybridFlowBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ZmqHybridFlowBackend")
    }
}

impl ZmqHybridFlowBackend {
    /// Bind `outbound` (PUB) and connect `inbound` (SUB).
    ///
    /// The two configs must name different endpoints. `outbound` is the socket
    /// peers subscribe to for spikes and embeddings. `inbound` is the socket
    /// this backend subscribes to for gradients and traces; the peer publisher
    /// binds that endpoint.
    pub fn open(
        outbound: ZmqTransportConfig,
        inbound: ZmqTransportConfig,
    ) -> Result<Self, ZmqMessageError> {
        let outbound = ZmqMessagePublisher::bind(outbound)?;
        let inbound = ZmqMessageSubscriber::connect(inbound)?;
        Ok(Self { outbound, inbound })
    }

    /// Close both sockets. Later trait calls return [`BackendError::CommunicationError`].
    pub fn close(&self) {
        self.outbound.close();
        self.inbound.close();
    }

    fn send_message(&self, message: IpcMessage) -> Result<(), BackendError> {
        self.outbound
            .send(&message)
            .map_err(transport::map_transport_error)
    }

    fn poll_inbound(&self) -> Result<Option<IpcMessage>, BackendError> {
        match self.inbound.try_recv() {
            Ok(message) => Ok(Some(message)),
            Err(ZmqMessageError::WouldBlock) => Ok(None),
            Err(err) => Err(transport::map_transport_error(err)),
        }
    }
}

impl HybridFlowBackend for ZmqHybridFlowBackend {
    fn send_spikes(&mut self, spikes: SpikeBatch) -> Result<(), BackendError> {
        self.send_message(IpcMessage::Spikes(spikes))
    }

    fn send_embeddings(&mut self, embeddings: EmbeddingBatch) -> Result<(), BackendError> {
        self.send_message(IpcMessage::Embeddings(embeddings))
    }

    fn try_receive_gradients(&mut self) -> Result<Option<GradientBatch>, BackendError> {
        match self.poll_inbound()? {
            None => Ok(None),
            Some(IpcMessage::GradientUpdate(batch)) => Ok(Some(batch)),
            Some(other) => Err(unexpected_variant(&other, "GradientUpdate")),
        }
    }

    fn try_receive_traces(&mut self) -> Result<Option<TraceBatch>, BackendError> {
        match self.poll_inbound()? {
            None => Ok(None),
            Some(IpcMessage::EligibilityTraces(batch)) => Ok(Some(batch)),
            Some(other) => Err(unexpected_variant(&other, "EligibilityTraces")),
        }
    }
}

fn unexpected_variant(message: &IpcMessage, expected: &'static str) -> BackendError {
    BackendError::Protocol(ProtocolError::UnexpectedVariant {
        found: variant_name(message),
        expected,
    })
}

fn variant_name(message: &IpcMessage) -> &'static str {
    if let Some(name) = payload_variant(message) {
        return name;
    }
    control_variant(message).expect("every IpcMessage variant has a name")
}

fn payload_variant(message: &IpcMessage) -> Option<&'static str> {
    match message {
        IpcMessage::Spikes(_) => Some("Spikes"),
        IpcMessage::Embeddings(_) => Some("Embeddings"),
        IpcMessage::Stimuli(_) => Some("Stimuli"),
        IpcMessage::Neuromodulators(_) => Some("Neuromodulators"),
        IpcMessage::Loss(_) => Some("Loss"),
        IpcMessage::ConfigUpdate(_) => Some("ConfigUpdate"),
        IpcMessage::GradientUpdate(_)
        | IpcMessage::EligibilityTraces(_)
        | IpcMessage::TrainingComplete
        | IpcMessage::Shutdown
        | IpcMessage::Ping => None,
    }
}

fn control_variant(message: &IpcMessage) -> Option<&'static str> {
    match message {
        IpcMessage::GradientUpdate(_) => Some("GradientUpdate"),
        IpcMessage::EligibilityTraces(_) => Some("EligibilityTraces"),
        IpcMessage::TrainingComplete => Some("TrainingComplete"),
        IpcMessage::Shutdown => Some("Shutdown"),
        IpcMessage::Ping => Some("Ping"),
        IpcMessage::Spikes(_)
        | IpcMessage::Embeddings(_)
        | IpcMessage::Stimuli(_)
        | IpcMessage::Neuromodulators(_)
        | IpcMessage::Loss(_)
        | IpcMessage::ConfigUpdate(_) => None,
    }
}

const _: fn() = assert_hybrid_send_sync;

fn assert_hybrid_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_hybrid<T: HybridFlowBackend>() {}

    assert_send::<ZmqHybridFlowBackend>();
    assert_sync::<ZmqHybridFlowBackend>();
    assert_hybrid::<ZmqHybridFlowBackend>();

    <crate::ZmqIpcBackend as not_hybrid::IfHybridThenAmbiguous<_>>::witness();
}

/// Overlapping-impl witness that `ZmqIpcBackend` is not a [`HybridFlowBackend`].
mod not_hybrid {
    use crate::HybridFlowBackend;

    pub trait IfHybridThenAmbiguous<Marker> {
        fn witness() {}
    }

    pub struct NotHybrid;
    pub struct Hybrid;

    impl<T: ?Sized> IfHybridThenAmbiguous<NotHybrid> for T {}
    impl<T: ?Sized + HybridFlowBackend> IfHybridThenAmbiguous<Hybrid> for T {}
}

#[cfg(test)]
mod coverage {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::{
        ConfigPayload, ConfigValue, EmbeddingBatch, GradientBatch, GradientUpdate,
        NeuromodulatorSnapshot, SpikeBatch, SpikeEvent, StimulusBatch, TraceBatch, TraceData,
    };

    fn endpoint() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "ipc:///tmp/corpus_ipc_zmq_hybrid_cov_{}_{}.ipc",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn named_messages() -> Vec<(&'static str, IpcMessage)> {
        let mut config = std::collections::HashMap::new();
        config.insert("gain".to_string(), ConfigValue::Float(0.5));
        vec![
            (
                "Spikes",
                IpcMessage::Spikes(SpikeBatch {
                    session_id: None,
                    batch_id: 1,
                    timestamp: 0,
                    spikes: vec![SpikeEvent {
                        channel: 1,
                        time: 1,
                        strength: 0.5,
                    }],
                    metadata: None,
                }),
            ),
            (
                "Embeddings",
                IpcMessage::Embeddings(EmbeddingBatch {
                    session_id: None,
                    batch_id: 1,
                    embedding: vec![0.5],
                    sequence_length: 1,
                }),
            ),
            (
                "Stimuli",
                IpcMessage::Stimuli(StimulusBatch {
                    session_id: None,
                    batch_id: 1,
                    timestamp: 0,
                    values: vec![0.5],
                    valid_mask: None,
                    metadata: None,
                }),
            ),
            (
                "Neuromodulators",
                IpcMessage::Neuromodulators(NeuromodulatorSnapshot {
                    tick: 1,
                    dopamine: 0.25,
                    cortisol: 0.5,
                    acetylcholine: 0.75,
                    tempo: 1.0,
                }),
            ),
            ("Loss", IpcMessage::Loss(0.25)),
            (
                "ConfigUpdate",
                IpcMessage::ConfigUpdate(ConfigPayload {
                    session_id: None,
                    config,
                }),
            ),
            (
                "GradientUpdate",
                IpcMessage::GradientUpdate(GradientBatch {
                    session_id: "session".to_string(),
                    batch_id: 1,
                    gradients: vec![GradientUpdate {
                        layer_id: "layer".to_string(),
                        gradients: vec![0.5],
                        eligibility_trace: None,
                    }],
                }),
            ),
            (
                "EligibilityTraces",
                IpcMessage::EligibilityTraces(TraceBatch {
                    session_id: "session".to_string(),
                    batch_id: 1,
                    traces: vec![TraceData {
                        channel_id: 1,
                        trace_value: 0.25,
                        last_spike_time: 1,
                    }],
                }),
            ),
            ("TrainingComplete", IpcMessage::TrainingComplete),
            ("Shutdown", IpcMessage::Shutdown),
            ("Ping", IpcMessage::Ping),
        ]
    }

    #[test]
    fn every_variant_name_is_stable() {
        assert_hybrid_send_sync();
        for (expected, message) in named_messages() {
            assert_eq!(variant_name(&message), expected);
            assert_eq!(
                payload_variant(&message).or(control_variant(&message)),
                Some(expected)
            );
        }
    }

    fn open_pair() -> (ZmqHybridFlowBackend, ZmqMessagePublisher, String) {
        let outbound = endpoint();
        let inbound = endpoint();
        let peer = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&inbound)).unwrap();
        let backend = ZmqHybridFlowBackend::open(
            ZmqTransportConfig::new(&outbound),
            ZmqTransportConfig::new(&inbound),
        )
        .unwrap();
        (backend, peer, outbound)
    }

    fn take_ready<T>(
        result: Result<Option<T>, BackendError>,
    ) -> Option<Result<Option<T>, BackendError>> {
        match result {
            Ok(None) => {
                std::thread::sleep(Duration::from_millis(1));
                None
            }
            other => Some(other),
        }
    }

    fn publish_until_gradient_outcome(
        backend: &mut ZmqHybridFlowBackend,
        peer: &ZmqMessagePublisher,
        message: &IpcMessage,
    ) -> Result<Option<GradientBatch>, BackendError> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            peer.send(message).unwrap();
            if let Some(outcome) = take_ready(backend.try_receive_gradients()) {
                return outcome;
            }
        }
        panic!("timed out waiting for a gradient outcome");
    }

    fn publish_until_trace_outcome(
        backend: &mut ZmqHybridFlowBackend,
        peer: &ZmqMessagePublisher,
        message: &IpcMessage,
    ) -> Result<Option<TraceBatch>, BackendError> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            peer.send(message).unwrap();
            if let Some(outcome) = take_ready(backend.try_receive_traces()) {
                return outcome;
            }
        }
        panic!("timed out waiting for a trace outcome");
    }

    #[test]
    fn unexpected_inbound_variants_are_protocol_errors() {
        let (mut backend, peer, _outbound) = open_pair();
        assert_eq!(format!("{backend:?}"), "ZmqHybridFlowBackend");

        let ping =
            publish_until_gradient_outcome(&mut backend, &peer, &IpcMessage::Ping).unwrap_err();
        assert!(matches!(
            ping,
            BackendError::Protocol(ProtocolError::UnexpectedVariant {
                found: "Ping",
                expected: "GradientUpdate",
            })
        ));

        let trace_ping =
            publish_until_trace_outcome(&mut backend, &peer, &IpcMessage::Ping).unwrap_err();
        assert!(matches!(
            trace_ping,
            BackendError::Protocol(ProtocolError::UnexpectedVariant {
                found: "Ping",
                expected: "EligibilityTraces",
            })
        ));
    }

    #[test]
    fn open_reports_a_failed_inbound_connect() {
        let outbound = endpoint();
        let err = ZmqHybridFlowBackend::open(
            ZmqTransportConfig::new(&outbound),
            ZmqTransportConfig::new("not-an-endpoint"),
        )
        .unwrap_err();
        assert!(matches!(err, ZmqMessageError::Setup(_)), "{err}");
    }
}
