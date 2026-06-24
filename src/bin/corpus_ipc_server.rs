//! Axum-based microservice exposing the `corpus-ipc` crate as a REST API.

use std::net::SocketAddr;
use tokio::net::TcpListener;
use std::sync::Arc;
use std::sync::Mutex;

use axum::{
    extract::Extension,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use corpus_ipc::{
    BackendError, BackendType, NeuralBackend,
};
use corpus_ipc::trait_def::BackendFactory;

type SharedBackend = Arc<Mutex<Box<dyn NeuralBackend>>>;

#[tokio::main]
async fn main() {
    // Select backend type via env var; default to Rust.
    let backend_type = match std::env::var("SPIKENAUT_BACKEND_TYPE").as_deref() {
        Ok("zmq") => {
            #[cfg(feature = "zmq")]
            { BackendType::ZmqBrain }
            #[cfg(not(feature = "zmq"))]
            {
                eprintln!("[corpus-ipc-service] 'zmq' backend requested but crate was built without 'zmq' feature; falling back to Rust backend");
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
    let addr: SocketAddr = std::env::var("SPIKENAUT_BIND")
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
    match be.process_signals(&payload.inputs) {
        Ok(out) => Ok(Json(ProcessRes { output: out })),
        Err(e) => Err(Json(SimpleRes { ok: false, message: e.to_string() })),
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
    reply(res, "reset")
}

// ---------- Helpers ----------

fn reply(res: Result<(), BackendError>, success_msg: &str) -> Json<SimpleRes> {
    match res {
        Ok(_) => Json(SimpleRes { ok: true, message: success_msg.into() }),
        Err(e) => Json(SimpleRes { ok: false, message: e.to_string() }),
    }
}
