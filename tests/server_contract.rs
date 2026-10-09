// SPDX-License-Identifier: MIT OR Apache-2.0

//! Black-box contract tests for the v0.2 REST router (LIM-1329).
//!
//! Drives the real `corpus_ipc::server::router` in-process with Tower's
//! `oneshot` — no socket binding required. Assertions cover success bodies,
//! every documented error class, malformed JSON, and the reset →
//! re-initialize lifecycle. Run with:
//!
//! ```bash
//! cargo test --locked --features server --test server_contract
//! ```

use std::sync::Arc;

use axum::response::IntoResponse;
use corpus_ipc::server::{AppState, router};
use corpus_ipc::{BackendError, BackendFactory, BackendType, IpcBackend};
use serde_json::{Value, json};
use tower::ServiceExt;

type TestResult = Result<(), Box<dyn std::error::Error>>;
type TestResponse = (u16, Value);

fn json_request(path: &str, body: &str) -> axum::http::Request<axum::body::Body> {
    axum::http::Request::post(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_owned()))
        .expect("request build")
}

async fn call(app: axum::Router, request: axum::http::Request<axum::body::Body>) -> TestResponse {
    let response = app.oneshot(request).await.expect("router oneshot");
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body collect");
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
        panic!(
            "non-JSON body ({status}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, body)
}

async fn post_json(app: axum::Router, path: &str, body: &str) -> TestResponse {
    call(app, json_request(path, body)).await
}

fn rust_router() -> axum::Router {
    router(BackendFactory::create(BackendType::Rust))
}

fn assert_error_envelope(status: u16, body: &Value, expected_status: u16, code: &str) {
    assert_eq!(status, expected_status, "body: {body}");
    assert_eq!(body["ok"], false);
    assert_eq!(body["error"]["code"], code);
    assert!(body["error"]["message"].is_string());
}

// ---------- Success envelopes ----------

#[tokio::test]
async fn initialize_returns_stable_success_body() -> TestResult {
    let (status, body) = post_json(rust_router(), "/initialize", "{}").await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body, json!({"ok": true, "message": "initialized"}));
    Ok(())
}

#[tokio::test]
async fn process_returns_ok_and_double_length_output() -> TestResult {
    let app = rust_router();
    post_json(app.clone(), "/initialize", "{}").await;
    let (status, body) = post_json(app, "/process", r#"{"inputs": [0.5, -0.25]}"#).await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["ok"], true);
    assert_eq!(body["output"], json!([0.5, 0.0, 0.0, 0.25]));
    Ok(())
}

#[tokio::test]
async fn save_state_and_reset_return_ack_bodies() -> TestResult {
    let app = rust_router();
    post_json(app.clone(), "/initialize", "{}").await;
    let (status, body) =
        post_json(app.clone(), "/save_state", r#"{"model_path": "state.bin"}"#).await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body, json!({"ok": true, "message": "state saved"}));
    let (status, body) = post_json(app, "/reset", "{}").await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body, json!({"ok": true, "message": "reset"}));
    Ok(())
}

// ---------- Error classes ----------

#[tokio::test]
async fn malformed_json_maps_to_400_bad_request() -> TestResult {
    for path in ["/initialize", "/process", "/save_state"] {
        let (status, body) = post_json(rust_router(), path, "{not json").await;
        assert_error_envelope(status, &body, 400, "bad_request");
    }
    Ok(())
}

#[tokio::test]
async fn missing_content_type_maps_to_415() -> TestResult {
    let request = axum::http::Request::post("/initialize")
        .body(axum::body::Body::from("{}"))
        .expect("request build");
    let (status, body) = call(rust_router(), request).await;
    assert_error_envelope(status, &body, 415, "unsupported_media_type");
    Ok(())
}

#[tokio::test]
async fn schema_validation_errors_map_to_422_invalid_input() -> TestResult {
    let app = rust_router();
    // Missing required field.
    let (status, body) = post_json(app.clone(), "/save_state", "{}").await;
    assert_error_envelope(status, &body, 422, "invalid_input");
    // Wrong field type.
    let (status, body) = post_json(app.clone(), "/process", r#"{"inputs": "not-an-array"}"#).await;
    assert_error_envelope(status, &body, 422, "invalid_input");
    // Wrong top-level type.
    let (status, body) = post_json(app, "/initialize", "[1, 2]").await;
    assert_error_envelope(status, &body, 422, "invalid_input");
    Ok(())
}

#[tokio::test]
async fn unsupported_method_maps_to_405() -> TestResult {
    let request = axum::http::Request::get("/process")
        .body(axum::body::Body::empty())
        .expect("request build");
    let response = rust_router().oneshot(request).await.expect("oneshot");
    assert_eq!(response.status().as_u16(), 405);
    Ok(())
}

// ---------- Reset / re-initialize lifecycle (409 gate) ----------

#[tokio::test]
async fn process_and_save_before_initialize_map_to_409() -> TestResult {
    let app = rust_router();
    let (status, body) = post_json(app.clone(), "/process", r#"{"inputs": [1.0]}"#).await;
    assert_error_envelope(status, &body, 409, "not_initialized");
    let (status, body) = post_json(app, "/save_state", r#"{"model_path": "state.bin"}"#).await;
    assert_error_envelope(status, &body, 409, "not_initialized");
    Ok(())
}

#[tokio::test]
async fn reset_requires_reinitialize_before_next_process() -> TestResult {
    let app = rust_router();
    post_json(app.clone(), "/initialize", "{}").await;
    post_json(app.clone(), "/process", r#"{"inputs": [1.0]}"#).await;
    let (status, _) = post_json(app.clone(), "/reset", "{}").await;
    assert_eq!(status, 200);
    // After reset the lifecycle gate demands a fresh /initialize.
    let (status, body) = post_json(app.clone(), "/process", r#"{"inputs": [1.0]}"#).await;
    assert_error_envelope(status, &body, 409, "not_initialized");
    let (status, _) = post_json(app.clone(), "/initialize", "{}").await;
    assert_eq!(status, 200);
    let (status, body) = post_json(app, "/process", r#"{"inputs": [2.0]}"#).await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["ok"], true);
    Ok(())
}

// ---------- Backend failure mapping ----------

/// Backend whose lifecycle succeeds but whose operations fail with distinct
/// `BackendError` classes, so the 422/502 mappings are exercised end-to-end.
struct ErrBackend;

impl IpcBackend for ErrBackend {
    fn process_batch(&mut self, _inputs: &[f32]) -> Result<Vec<f32>, BackendError> {
        Err(BackendError::InvalidInput("bad batch".into()))
    }
    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        Ok(())
    }
    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        Err(BackendError::CommunicationError(
            "engine unreachable".into(),
        ))
    }
    fn get_spike_states(&self) -> Vec<bool> {
        Vec::new()
    }
    fn reset(&mut self) -> Result<(), BackendError> {
        Err(BackendError::ModelError("io failure".into()))
    }
}

fn err_router() -> axum::Router {
    router(Box::new(ErrBackend))
}

#[tokio::test]
async fn backend_invalid_input_maps_to_422() -> TestResult {
    let app = err_router();
    post_json(app.clone(), "/initialize", "{}").await;
    let (status, body) = post_json(app, "/process", r#"{"inputs": [1.0]}"#).await;
    assert_error_envelope(status, &body, 422, "invalid_input");
    Ok(())
}

#[tokio::test]
async fn backend_operational_failures_map_to_502() -> TestResult {
    let app = err_router();
    post_json(app.clone(), "/initialize", "{}").await;
    let (status, body) =
        post_json(app.clone(), "/save_state", r#"{"model_path": "state.bin"}"#).await;
    assert_error_envelope(status, &body, 502, "backend_error");
    let (status, body) = post_json(app, "/reset", "{}").await;
    assert_error_envelope(status, &body, 502, "backend_error");
    Ok(())
}

#[tokio::test]
async fn poisoned_backend_lock_maps_to_500() -> TestResult {
    let state = Arc::new(AppState::new(Box::new(ErrBackend)));
    let poisoned = Arc::clone(&state);
    std::thread::spawn(move || {
        let _guard = poisoned.lock_backend().expect("lock");
        panic!("poison the mutex");
    })
    .join()
    .expect_err("poisoning thread panics");
    let error = state
        .lock_backend()
        .err()
        .expect("poisoned mutex must surface an error");
    let response = error.into_response();
    assert_eq!(response.status().as_u16(), 500);
    Ok(())
}

// ---------- ZMQ backend router exercise (feature `zmq`) ----------

#[cfg(feature = "zmq")]
#[tokio::test]
async fn zmq_router_enforces_lifecycle_gate() -> TestResult {
    #[allow(deprecated)]
    let app = router(BackendFactory::create(BackendType::ZmqIpc));
    // The 409 gate is transport-independent.
    let (status, body) = post_json(app.clone(), "/process", r#"{"inputs": [1.0]}"#).await;
    assert_error_envelope(status, &body, 409, "not_initialized");
    // Reset is idempotent even before initialize.
    let (status, body) = post_json(app.clone(), "/reset", "{}").await;
    assert_eq!(status, 200, "body: {body}");
    // Connecting the SUB socket does not require a live publisher; the
    // subsequent /process drains to EAGAIN and returns the empty cache.
    let (status, body) = post_json(app.clone(), "/initialize", "{}").await;
    assert_eq!(status, 200, "body: {body}");
    let (status, body) = post_json(app, "/process", r#"{"inputs": [1.0]}"#).await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["ok"], true);
    Ok(())
}
