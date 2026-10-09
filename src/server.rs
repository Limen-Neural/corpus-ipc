// SPDX-License-Identifier: MIT OR Apache-2.0

//! Axum-based REST microservice exposing an [`IpcBackend`] over HTTP.
//!
//! Built only when the `server` crate feature is enabled. Add `zmq` as well if
//! the process should be able to select `CORPUS_IPC_BACKEND_TYPE=zmq`.
//!
//! # v0.2 contract
//!
//! The stable request/response/status table lives in `docs/rest-api.md`
//! (shipped in the crates.io package). Summary:
//!
//! - `POST /initialize` — initialize the backend (`{"model_path": string?}`).
//! - `POST /process` — run a batch (`{"inputs": [f32]}`), requires prior
//!   successful `/initialize`.
//! - `POST /save_state` — persist state (`{"model_path": string}`), requires
//!   prior successful `/initialize`.
//! - `POST /reset` — reset state; connection-oriented backends drop their
//!   transport and require `/initialize` again before `/process`.
//!
//! Every error returns the shared envelope
//! `{"ok": false, "error": {"code": ..., "message": ...}}` with an explicit
//! status class: 400 malformed JSON, 415 missing/wrong content type, 422
//! validation/input errors, 409 use before initialize, 502 backend failures,
//! 500 internal failures.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, routing::post};
use serde::{Deserialize, Serialize};

use crate::{BackendError, IpcBackend};

/// Shareable REST application state: one backend behind a mutex plus the
/// server-observed initialized flag.
///
/// The flag is authoritative for the 409 gate: it is set only after a
/// successful `initialize` and cleared after a successful `reset`, matching
/// the `IpcBackend` reset contract (connection-oriented backends drop their
/// initialized state on reset).
pub struct AppState {
    backend: Mutex<Box<dyn IpcBackend>>,
    initialized: AtomicBool,
}

impl AppState {
    /// Wrap a backend instance for the router. Starts uninitialized.
    pub fn new(backend: Box<dyn IpcBackend>) -> Self {
        Self {
            backend: Mutex::new(backend),
            initialized: AtomicBool::new(false),
        }
    }

    /// Lock the shared backend. Surfaces a 500 `internal_error` on poison.
    pub fn lock_backend(&self) -> Result<MutexGuard<'_, Box<dyn IpcBackend>>, ApiError> {
        self.backend
            .lock()
            .map_err(|_| ApiError::internal("backend mutex poisoned"))
    }
}

/// Build the v0.2 REST router over the given backend.
///
/// In-process equivalent of the `corpus_ipc_server` binary's routes; callers
/// may drive it directly via `tower::ServiceExt` without binding a socket.
pub fn router(backend: Box<dyn IpcBackend>) -> Router {
    Router::new()
        .route("/initialize", post(initialize))
        .route("/process", post(process))
        .route("/save_state", post(save_state))
        .route("/reset", post(reset))
        .with_state(Arc::new(AppState::new(backend)))
}

// ---------- Request/response types ----------

#[derive(Debug, Deserialize)]
struct InitializeReq {
    model_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProcessReq {
    inputs: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SaveStateReq {
    model_path: String,
}

#[derive(Debug, Serialize)]
struct AckRes {
    ok: bool,
    message: &'static str,
}

#[derive(Debug, Serialize)]
struct ProcessRes {
    ok: bool,
    output: Vec<f32>,
}

/// Stable v0.2 error envelope body.
#[derive(Debug, Serialize)]
struct ErrorRes {
    ok: bool,
    error: ErrorInfo,
}

#[derive(Debug, Serialize)]
struct ErrorInfo {
    code: &'static str,
    message: String,
}

// ---------- Errors ----------

/// HTTP-mapped API error: a stable `code`, an HTTP status class, and a
/// human-readable `message` (never serialized as the wire `code`).
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    fn not_initialized() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "not_initialized",
            "backend not initialized — call POST /initialize first",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorRes {
                ok: false,
                error: ErrorInfo {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response()
    }
}

/// Map a [`JsonRejection`] to the v2 error classes: syntax/framing problems are
/// 400 `bad_request`, a missing JSON content type is 415
/// `unsupported_media_type`, and deserialization/validation failures are 422
/// `invalid_input`.
fn json_rejection(rejection: JsonRejection) -> ApiError {
    let (status, code) = match rejection {
        JsonRejection::JsonSyntaxError(_) => (StatusCode::BAD_REQUEST, "bad_request"),
        JsonRejection::MissingJsonContentType(_) => {
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type")
        }
        JsonRejection::JsonDataError(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
        _ => (StatusCode::BAD_REQUEST, "bad_request"),
    };
    ApiError::new(status, code, rejection.to_string())
}

/// Map a [`BackendError`] to the v2 error classes: caller input faults are 422
/// `invalid_input`; every backend operational failure is 502 `backend_error`.
fn backend_error(error: &BackendError) -> ApiError {
    let (status, code) = match error {
        BackendError::InvalidInput(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
        _ => (StatusCode::BAD_GATEWAY, "backend_error"),
    };
    ApiError::new(status, code, error.to_string())
}

type ApiResult<T> = Result<T, ApiError>;

/// Gate for endpoints that require a live initialized session, then run `f`
/// against the locked backend. Shared by `/process` and `/save_state`.
fn with_initialized_backend<R>(
    state: &AppState,
    f: impl FnOnce(&mut Box<dyn IpcBackend>) -> Result<R, BackendError>,
) -> ApiResult<R> {
    let mut backend = state.lock_backend()?;
    // Check the flag under the backend lock: /reset and /initialize update it
    // while holding the same lock, so gate and operation stay serialized and
    // a concurrent reset cannot sneak between the check and the call.
    if !state.initialized.load(Ordering::SeqCst) {
        return Err(ApiError::not_initialized());
    }
    f(&mut backend).map_err(|e| backend_error(&e))
}

// ---------- Handlers ----------

async fn initialize(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<InitializeReq>, JsonRejection>,
) -> ApiResult<Json<AckRes>> {
    let Json(payload) = payload.map_err(json_rejection)?;
    let mut backend = state.lock_backend()?;
    backend
        .initialize(payload.model_path.as_deref())
        .map_err(|e| backend_error(&e))?;
    state.initialized.store(true, Ordering::SeqCst);
    Ok(Json(AckRes {
        ok: true,
        message: "initialized",
    }))
}

async fn process(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ProcessReq>, JsonRejection>,
) -> ApiResult<Json<ProcessRes>> {
    let Json(payload) = payload.map_err(json_rejection)?;
    let output =
        with_initialized_backend(&state, |backend| backend.process_batch(&payload.inputs))?;
    Ok(Json(ProcessRes { ok: true, output }))
}

async fn save_state(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<SaveStateReq>, JsonRejection>,
) -> ApiResult<Json<AckRes>> {
    let Json(payload) = payload.map_err(json_rejection)?;
    with_initialized_backend(&state, |backend| backend.save_state(&payload.model_path))?;
    Ok(Json(AckRes {
        ok: true,
        message: "state saved",
    }))
}

async fn reset(State(state): State<Arc<AppState>>) -> ApiResult<Json<AckRes>> {
    let mut backend = state.lock_backend()?;
    backend.reset().map_err(|e| backend_error(&e))?;
    // Both shipped backends clear their initialized state on reset; the next
    // /process must re-initialize (matching the IpcBackend reset contract for
    // connection-oriented backends).
    state.initialized.store(false, Ordering::SeqCst);
    Ok(Json(AckRes {
        ok: true,
        message: "reset",
    }))
}
