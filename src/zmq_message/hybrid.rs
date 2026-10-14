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
    match message {
        IpcMessage::Spikes(_) => "Spikes",
        IpcMessage::Embeddings(_) => "Embeddings",
        IpcMessage::Stimuli(_) => "Stimuli",
        IpcMessage::Neuromodulators(_) => "Neuromodulators",
        IpcMessage::Loss(_) => "Loss",
        IpcMessage::ConfigUpdate(_) => "ConfigUpdate",
        IpcMessage::GradientUpdate(_) => "GradientUpdate",
        IpcMessage::EligibilityTraces(_) => "EligibilityTraces",
        IpcMessage::TrainingComplete => "TrainingComplete",
        IpcMessage::Shutdown => "Shutdown",
        IpcMessage::Ping => "Ping",
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

    let _ = <crate::ZmqIpcBackend as not_hybrid::IfHybridThenAmbiguous<_>>::witness;
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
