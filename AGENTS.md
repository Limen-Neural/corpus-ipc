# corpus-ipc

Rust IPC library for bridging to external compute engines via ZeroMQ.
Part of the Limen-Neural neuromorphic computing stack.

## Build & Test

```bash
cargo build              # compile all features
cargo test               # run unit tests
cargo build --features zmq  # compile with ZMQ backend
```

## Architecture

- `src/trait_def.rs` — RuntimeBackend trait (core abstraction)
- `src/models.rs` — wire protocol types (RuntimeSnapshot, RuntimeMessage, SpikeBatch, etc.)
- `src/zmq_backend.rs` — ZMQ SUB backend (feature-gated `zmq`)
- `src/rust_backend.rs` — pure-Rust native backend
- `src/error.rs` — BackendError types
- `src/bin/corpus_ipc_server.rs` — HTTP server binary (axum + tokio)

## Conventions

- Edition 2024, dual MIT/Apache-2.0 license
- `#[must_use]` on constructors, `const fn` where possible
- Use `Ok(())` not `Ok(_)` for unit returns
- serde aliases for wire-protocol backward compatibility
- Do NOT rename literature citations in doc comments
- `docs/plans/` is gitignored — do not track planning documents

## CI

```bash
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-features && cargo test --all-features
```
