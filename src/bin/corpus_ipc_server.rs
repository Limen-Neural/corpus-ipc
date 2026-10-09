// SPDX-License-Identifier: MIT OR Apache-2.0

//! Axum-based microservice exposing the `corpus-ipc` crate as a REST API.
//!
//! Built only when the `server` crate feature is enabled
//! (`required-features = ["server"]`). Add `zmq` as well if the process should
//! be able to select `CORPUS_IPC_BACKEND_TYPE=zmq`.
//!
//! The v0.2 request/response/error contract is documented in
//! `docs/rest-api.md`.

use std::net::SocketAddr;
use tokio::net::TcpListener;

use corpus_ipc::BackendType;
use corpus_ipc::trait_def::BackendFactory;

#[tokio::main]
async fn main() {
    // Select backend type via env var (CORPUS_IPC_BACKEND_TYPE); default to Rust.
    let backend_type = match std::env::var("CORPUS_IPC_BACKEND_TYPE").as_deref() {
        Ok("zmq") => {
            #[cfg(feature = "zmq")]
            {
                BackendType::ZmqIpc
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

    let app = corpus_ipc::server::router(BackendFactory::create(backend_type));

    // Loopback by default. Set CORPUS_IPC_BIND for another address; remote
    // binds need an external access-control boundary (no auth/TLS here).
    let addr: SocketAddr = std::env::var("CORPUS_IPC_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8080".into())
        .parse()
        .expect("invalid bind address");

    println!("[corpus-ipc-service] listening on {addr}");
    let listener = TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
