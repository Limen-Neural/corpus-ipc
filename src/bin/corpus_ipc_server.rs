// SPDX-License-Identifier: MIT OR Apache-2.0

//! Axum-based microservice exposing the `corpus-ipc` crate as a REST API.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::net::TcpListener;

use axum::{Json, Router, extract::Extension, routing::post};
use corpus_ipc::trait_def::BackendFactory;
use corpus_ipc::{BackendError, BackendType, RuntimeBackend};
use serde::{Deserialize, Serialize};

type SharedBackend = Arc<Mutex<Box<dyn RuntimeBackend>>>;

#[tokio::main]
async fn main() {
    // Select backend type via env var (CORPUS_IPC_BACKEND_TYPE); default to Rust.
    // Hard rename pass for service entrypoint - legacy fallback not kept
    // per cleanup goals (addresses bot feedback on rename completeness).
    let backend_type = match std::env::var("CORPUS_IPC_BACKEND_TYPE").as_deref() {
        Ok("zmq") => {
            #[cfg(feature = "zmq")]
            {
                BackendType::ZmqRuntime
            }
            #[cfg(not(feature = "zmq"))]
            {
                eprintln!(
                    "[corpus-ipc-service] 'zmq' backend requested but crate was built without 'zmq' feature; falling back to Rust backend"
                );
                BackendType::Rust
            }
        }
        _ => BackendType::Rust,
    };

    // Instantiate backend and wrap in a thread-safe container.
    let backend: SharedBackend = Arc::new(Mutex::new(BackendFactory::create(backend_type)));

    // Build routes.
    let app = Router::new()
        .route("/initialize", post(initialize))
        .route("/process", post(process))
        .route("/save_state", post(save_state))
        .route("/reset", post(reset))
        .layer(Extension(backend));

    // Bind address (0.0.0.0:8080 by default).
    // Hard rename pass: only CORPUS_IPC_BIND used (legacy fallback removed per goals).
    let addr: SocketAddr = std::env::var("CORPUS_IPC_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()
        .expect("invalid bind address");

    println!("[corpus-ipc-service] listening on {addr}");
    let listener = TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
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

#[derive(Debug, Serialize)]
struct ProcessRes {
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SaveStateReq {
    model_path: String,
}

#[derive(Debug, Serialize)]
struct SimpleRes {
    ok: bool,
    message: String,
}

// ---------- Handlers ----------

async fn initialize(
    Extension(backend): Extension<SharedBackend>,
    Json(payload): Json<InitializeReq>,
) -> Json<SimpleRes> {
    let mut be = backend.lock().unwrap();
    let res = be.initialize(payload.model_path.as_deref());
    reply(res, "initialized")
}

async fn process(
    Extension(backend): Extension<SharedBackend>,
    Json(payload): Json<ProcessReq>,
) -> Result<Json<ProcessRes>, Json<SimpleRes>> {
    let mut be = backend.lock().unwrap();
    match be.process_batch(&payload.inputs) {
        Ok(out) => Ok(Json(ProcessRes { output: out })),
        Err(e) => Err(Json(SimpleRes {
            ok: false,
            message: e.to_string(),
        })),
    }
}

async fn save_state(
    Extension(backend): Extension<SharedBackend>,
    Json(payload): Json<SaveStateReq>,
) -> Json<SimpleRes> {
    let be = backend.lock().unwrap();
    let res = be.save_state(&payload.model_path);
    reply(res, "state saved")
}

async fn reset(Extension(backend): Extension<SharedBackend>) -> Json<SimpleRes> {
    let mut be = backend.lock().unwrap();
    let res = be.reset();
    // Note: after reset(), some backends (e.g. ZMQ) clear their connection state
    // and require a subsequent /initialize call before the next /process.
    // Clients should call /initialize (with model_path if needed) after /reset
    // if they intend to continue processing.
    reply(res, "reset")
}

// ---------- Helpers ----------

fn reply(res: Result<(), BackendError>, success_msg: &str) -> Json<SimpleRes> {
    match res {
        Ok(_) => Json(SimpleRes {
            ok: true,
            message: success_msg.into(),
        }),
        Err(e) => Json(SimpleRes {
            ok: false,
            message: e.to_string(),
        }),
    }
}
