// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::ValidationError;

fn endpoint() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "ipc:///tmp/corpus_ipc_zmq_cov_{}_{}.ipc",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn assert_invalid(config: &ZmqTransportConfig, expected: &str) {
    match config.check() {
        Err(ZmqMessageError::InvalidConfig(message)) => assert_eq!(message, expected),
        other => panic!("expected invalid config, got {other:?}"),
    }
}

#[test]
fn config_check_rejects_each_constraint() {
    assert_invalid(&ZmqTransportConfig::new(""), "endpoint must be non-empty");
    assert_invalid(
        &ZmqTransportConfig::new("ipc://bad\0endpoint"),
        "endpoint must not contain an interior null",
    );

    let mut config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.send_hwm = -1;
    assert_invalid(&config, "send_hwm must be >= 0");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.receive_hwm = -1;
    assert_invalid(&config, "receive_hwm must be >= 0");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.linger_ms = -2;
    assert_invalid(&config, "linger_ms must be >= -1");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.reconnect_ivl_ms = 0;
    assert_invalid(&config, "reconnect_ivl_ms must be > 0");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.reconnect_ivl_max_ms = 0;
    assert_invalid(&config, "reconnect_ivl_max_ms must be > 0");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.reconnect_ivl_ms = 200;
    config.reconnect_ivl_max_ms = 100;
    assert_invalid(&config, "reconnect_ivl_max_ms must be >= reconnect_ivl_ms");

    config = ZmqTransportConfig::new("tcp://127.0.0.1:9");
    config.send_hwm = 0;
    config.receive_hwm = 0;
    config.check().unwrap();
}

#[test]
fn oversized_encoded_frames_are_rejected_before_send() {
    assert!(ensure_frame_fits(0).is_ok());
    assert!(ensure_frame_fits(STRUCTURED_MAX_FRAME_BYTES as usize).is_ok());
    let err = ensure_frame_fits(usize::MAX).unwrap_err();
    assert!(
        matches!(err, ZmqMessageError::FrameTooLarge { .. }),
        "{err}"
    );
}

#[test]
fn send_and_transport_errors_map_without_a_socket() {
    assert!(map_send(Ok(())).is_ok());
    assert!(matches!(
        map_send(Err(zmq::Error::EAGAIN)),
        Err(ZmqMessageError::HighWaterMark)
    ));
    assert!(matches!(
        map_send(Err(zmq::Error::EINVAL)),
        Err(ZmqMessageError::Transport(_))
    ));

    let validation = ValidationError::non_finite("strength", f32::NAN);
    assert!(matches!(
        map_transport_error(ZmqMessageError::Validation(validation.clone())),
        crate::BackendError::InvalidInput(_)
    ));
    assert!(matches!(
        map_transport_error(ZmqMessageError::Encode(CanonicalEncodeError::Validation(
            validation
        ))),
        crate::BackendError::InvalidInput(_)
    ));
    assert!(matches!(
        map_transport_error(ZmqMessageError::WouldBlock),
        crate::BackendError::CommunicationError(_)
    ));
    assert!(matches!(
            map_transport_error(decode_frame(br#"{"payload":{"Neuromodulators":{"acetylcholine":0.0,"cortisol":0.0,"dopamine":5.0,"tempo":1.0,"tick":1}},"wire_version":1}"#).unwrap_err()),
            crate::BackendError::InvalidInput(_)
        ));
    assert!(matches!(
        map_transport_error(
            decode_frame(br#"{"payload":{"Neuromodulators":{"tick":1}},"wire_version":1}"#)
                .unwrap_err()
        ),
        crate::BackendError::CommunicationError(_)
    ));
}

#[test]
fn publisher_and_subscriber_debug_and_closed_options() {
    assert_socket_send_sync();
    let endpoint = endpoint();
    let publisher = ZmqMessagePublisher::bind(ZmqTransportConfig::new(&endpoint)).unwrap();
    let subscriber = ZmqMessageSubscriber::connect(ZmqTransportConfig::new(&endpoint)).unwrap();
    assert_eq!(format!("{publisher:?}"), "ZmqMessagePublisher");
    assert_eq!(format!("{subscriber:?}"), "ZmqMessageSubscriber");
    publisher.close();
    subscriber.close();
    assert!(matches!(
        publisher.socket_options_for_tests(),
        Err(ZmqMessageError::Closed)
    ));
    assert!(matches!(
        subscriber.socket_options_for_tests(),
        Err(ZmqMessageError::Closed)
    ));

    poison_publisher_lock(&publisher);
    assert!(matches!(
        publisher.send(&crate::IpcMessage::Ping),
        Err(ZmqMessageError::Closed)
    ));
    assert!(matches!(
        publisher.send(&crate::IpcMessage::Loss(f32::NAN)),
        Err(ZmqMessageError::Closed)
    ));
}

fn poison_publisher_lock(publisher: &ZmqMessagePublisher) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = publisher.socket.lock().expect("lock");
        panic!("poison the publisher mutex");
    }));
}

#[test]
fn bind_and_connect_report_setup_failures() {
    let bind_err =
        ZmqMessagePublisher::bind(ZmqTransportConfig::new("not-an-endpoint")).unwrap_err();
    assert!(bind_err.to_string().contains("bind"), "{bind_err}");

    let connect_err =
        ZmqMessageSubscriber::connect(ZmqTransportConfig::new("not-an-endpoint")).unwrap_err();
    assert!(connect_err.to_string().contains("connect"), "{connect_err}");
}
