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
/// The other supported inbound variant is held in a one-message slot and
/// returned by its own method. A second sibling while that slot is full
/// returns [`BackendError::CommunicationError`]; the new frame is consumed
/// and the held frame stays. Any unrelated inbound variant returns
/// [`BackendError::Protocol`] ([`ProtocolError::UnexpectedVariant`]). That
/// frame is consumed; it is not discarded quietly and it is not pushed back
/// onto the socket. Inbound validation failures surface as
/// [`BackendError::InvalidInput`].
///
/// This type does not implement [`crate::IpcBackend`]. Binary readouts stay on
/// [`crate::ZmqIpcBackend`].
///
/// Delivery is the publisher/subscriber contract: best-effort, at-most-once,
/// no acknowledgements. See the [module docs](crate::zmq_message).
pub struct ZmqHybridFlowBackend {
    outbound: ZmqMessagePublisher,
    inbound: ZmqMessageSubscriber,
    held_gradient: Option<GradientBatch>,
    held_trace: Option<TraceBatch>,
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
        if outbound.endpoint == inbound.endpoint {
            return Err(ZmqMessageError::InvalidConfig(
                "outbound and inbound endpoints must differ",
            ));
        }
        let outbound = ZmqMessagePublisher::bind(outbound)?;
        let inbound = ZmqMessageSubscriber::connect(inbound)?;
        Ok(Self {
            outbound,
            inbound,
            held_gradient: None,
            held_trace: None,
        })
    }

    /// Close both sockets. Later trait calls return [`BackendError::CommunicationError`],
    /// including a send whose payload would not validate.
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
        self.pull(Want::Gradient, false)?
            .map(as_gradient)
            .transpose()
    }

    fn try_receive_traces(&mut self) -> Result<Option<TraceBatch>, BackendError> {
        self.pull(Want::Trace, false)?.map(as_trace).transpose()
    }
}

#[derive(Clone, Copy)]
enum Want {
    Gradient,
    Trace,
}

fn as_gradient(message: IpcMessage) -> Result<GradientBatch, BackendError> {
    match message {
        IpcMessage::GradientUpdate(batch) => Ok(batch),
        other => Err(unexpected_variant(&other, "GradientUpdate")),
    }
}

fn as_trace(message: IpcMessage) -> Result<TraceBatch, BackendError> {
    match message {
        IpcMessage::EligibilityTraces(batch) => Ok(batch),
        other => Err(unexpected_variant(&other, "EligibilityTraces")),
    }
}

impl ZmqHybridFlowBackend {
    fn pull(&mut self, want: Want, extra: bool) -> Result<Option<IpcMessage>, BackendError> {
        if let Some(message) = self.take_pending(want) {
            return Ok(Some(message));
        }
        self.read_or_defer(want, extra)
    }

    fn take_pending(&mut self, want: Want) -> Option<IpcMessage> {
        match want {
            Want::Gradient => self.held_gradient.take().map(IpcMessage::GradientUpdate),
            Want::Trace => self.held_trace.take().map(IpcMessage::EligibilityTraces),
        }
    }

    fn read_or_defer(
        &mut self,
        want: Want,
        extra: bool,
    ) -> Result<Option<IpcMessage>, BackendError> {
        let Some(message) = self.poll_inbound()? else {
            return Ok(None);
        };
        self.keep(want, message, extra)
    }

    fn keep(
        &mut self,
        want: Want,
        message: IpcMessage,
        extra: bool,
    ) -> Result<Option<IpcMessage>, BackendError> {
        if is_wanted(want, &message) {
            return Ok(Some(message));
        }
        self.defer_sibling(message, want)?;
        self.after_defer(want, extra)
    }

    fn after_defer(&mut self, want: Want, extra: bool) -> Result<Option<IpcMessage>, BackendError> {
        if extra {
            return Ok(None);
        }
        self.read_or_defer(want, true)
    }

    fn defer_sibling(&mut self, message: IpcMessage, want: Want) -> Result<(), BackendError> {
        match (want, message) {
            (Want::Gradient, IpcMessage::EligibilityTraces(batch)) => {
                store_one(&mut self.held_trace, batch, "eligibility trace")
            }
            (Want::Trace, IpcMessage::GradientUpdate(batch)) => {
                store_one(&mut self.held_gradient, batch, "gradient")
            }
            (_, other) => Err(unexpected_variant(&other, wanted_name(want))),
        }
    }
}

fn is_wanted(want: Want, message: &IpcMessage) -> bool {
    matches!(
        (want, message),
        (Want::Gradient, IpcMessage::GradientUpdate(_))
            | (Want::Trace, IpcMessage::EligibilityTraces(_))
    )
}

fn wanted_name(want: Want) -> &'static str {
    match want {
        Want::Gradient => "GradientUpdate",
        Want::Trace => "EligibilityTraces",
    }
}

fn store_one<T>(slot: &mut Option<T>, value: T, label: &str) -> Result<(), BackendError> {
    if slot.is_some() {
        return Err(BackendError::CommunicationError(format!(
            "inbound {label} slot already holds a message"
        )));
    }
    *slot = Some(value);
    Ok(())
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

    use super::super::tests::{sample_gradients, sample_traces};
    use super::*;

    fn endpoint() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "ipc:///tmp/corpus_ipc_zmq_hybrid_cov_{}_{}.ipc",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn decoded(json: &str) -> IpcMessage {
        serde_json::from_str(json).unwrap_or_else(|err| panic!("{json}: {err}"))
    }

    #[test]
    fn every_variant_name_is_stable() {
        assert_hybrid_send_sync();
        let cases = [
            (
                "Spikes",
                r#"{"Spikes":{"session_id":null,"batch_id":1,"timestamp":0,"spikes":[{"channel":1,"time":1,"strength":0.5}],"metadata":null}}"#,
            ),
            (
                "Embeddings",
                r#"{"Embeddings":{"session_id":null,"batch_id":1,"embedding":[0.5],"sequence_length":1}}"#,
            ),
            (
                "Stimuli",
                r#"{"Stimuli":{"session_id":null,"batch_id":1,"timestamp":0,"values":[0.5],"valid_mask":null,"metadata":null}}"#,
            ),
            (
                "Neuromodulators",
                r#"{"Neuromodulators":{"tick":1,"dopamine":0.25,"cortisol":0.5,"acetylcholine":0.75,"tempo":1.0}}"#,
            ),
            ("Loss", r#"{"Loss":0.25}"#),
            (
                "ConfigUpdate",
                r#"{"ConfigUpdate":{"session_id":null,"config":{"gain":0.5}}}"#,
            ),
            (
                "GradientUpdate",
                r#"{"GradientUpdate":{"session_id":"session","batch_id":1,"gradients":[{"layer_id":"layer","gradients":[0.5],"eligibility_trace":null}]}}"#,
            ),
            (
                "EligibilityTraces",
                r#"{"EligibilityTraces":{"session_id":"session","batch_id":1,"traces":[{"channel_id":1,"trace_value":0.25,"last_spike_time":1}]}}"#,
            ),
            ("TrainingComplete", r#""TrainingComplete""#),
            ("Shutdown", r#""Shutdown""#),
            ("Ping", r#""Ping""#),
        ];
        for (expected, json) in cases {
            let message = decoded(json);
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

    fn quiet_gradients(backend: &mut ZmqHybridFlowBackend) {
        let started = Instant::now();
        let mut quiet_since = Instant::now();
        while started.elapsed() < Duration::from_millis(300) {
            if gradient_is_quiet(backend, &mut quiet_since) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn gradient_is_quiet(backend: &mut ZmqHybridFlowBackend, quiet_since: &mut Instant) -> bool {
        match backend.try_receive_gradients() {
            Ok(None) => quiet_since.elapsed() >= Duration::from_millis(40),
            Ok(Some(_)) => {
                *quiet_since = Instant::now();
                false
            }
            Err(err) => panic!("{err}"),
        }
    }

    fn publish_until<T>(
        backend: &mut ZmqHybridFlowBackend,
        peer: &ZmqMessagePublisher,
        message: &IpcMessage,
        poll: impl Fn(&mut ZmqHybridFlowBackend) -> Result<Option<T>, BackendError>,
    ) -> Result<Option<T>, BackendError> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            peer.send(message).unwrap();
            if let Some(outcome) = take_ready(poll(backend)) {
                return outcome;
            }
        }
        panic!("timed out waiting for an inbound outcome");
    }

    #[test]
    fn unexpected_inbound_variants_are_protocol_errors() {
        let (mut backend, peer, _outbound) = open_pair();
        assert_eq!(format!("{backend:?}"), "ZmqHybridFlowBackend");

        let ping = publish_until(&mut backend, &peer, &IpcMessage::Ping, |backend| {
            backend.try_receive_gradients()
        })
        .unwrap_err();
        assert!(matches!(
            ping,
            BackendError::Protocol(ProtocolError::UnexpectedVariant {
                found: "Ping",
                expected: "GradientUpdate",
            })
        ));

        let trace_ping = publish_until(&mut backend, &peer, &IpcMessage::Ping, |backend| {
            backend.try_receive_traces()
        })
        .unwrap_err();
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

    #[test]
    fn open_rejects_equal_endpoints() {
        let shared = endpoint();
        let err = ZmqHybridFlowBackend::open(
            ZmqTransportConfig::new(&shared),
            ZmqTransportConfig::new(shared),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ZmqMessageError::InvalidConfig("outbound and inbound endpoints must differ")
        ));
    }

    #[test]
    fn batch_converters_reject_ping() {
        assert!(matches!(
            as_gradient(IpcMessage::Ping),
            Err(BackendError::Protocol(_))
        ));
        assert!(matches!(
            as_trace(IpcMessage::Ping),
            Err(BackendError::Protocol(_))
        ));
    }

    #[test]
    fn a_full_sibling_slot_keeps_the_held_batch() {
        let mut slot = Some(sample_traces());
        let err = store_one(&mut slot, sample_traces(), "eligibility trace").unwrap_err();
        assert!(matches!(err, BackendError::CommunicationError(_)));
        assert!(slot.is_some());
    }

    #[test]
    fn a_trace_ahead_of_a_gradient_stays_available() {
        let (mut backend, peer, _outbound) = open_pair();
        let trace = IpcMessage::EligibilityTraces(sample_traces());
        let gradient = IpcMessage::GradientUpdate(sample_gradients());
        publish_until(&mut backend, &peer, &gradient, |backend| {
            backend.try_receive_gradients()
        })
        .unwrap()
        .unwrap();
        quiet_gradients(&mut backend);
        peer.send(&trace).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got_gradient = None;
        while Instant::now() < deadline && got_gradient.is_none() {
            peer.send(&gradient).unwrap();
            match backend.try_receive_gradients() {
                Ok(Some(batch)) => got_gradient = Some(batch),
                Ok(None) => std::thread::sleep(Duration::from_millis(1)),
                Err(err) => panic!("{err}"),
            }
        }
        assert_eq!(got_gradient, Some(sample_gradients()));
        match backend.try_receive_traces() {
            Ok(Some(batch)) => assert_eq!(batch, sample_traces()),
            Ok(None) => {
                let batch = publish_until(&mut backend, &peer, &trace, |backend| {
                    backend.try_receive_traces()
                })
                .unwrap();
                assert_eq!(batch, Some(sample_traces()));
            }
            Err(err) => panic!("{err}"),
        }
    }
}
