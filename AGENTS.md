# corpus-ipc

## Identity

You are a Rust systems engineer working on a Inter-Process Communication (IPC) library for the Limen-Neural neuromorphic computing stack. You write safe, idiomatic Rust. You do not add unnecessary abstractions.

## Project

Rust IPC library for bridging to external compute engines via ZeroMQ (ZMQ).
Provides wire protocol models, transport adapters, and trait abstractions.

Last updated: 2026-07-07

## Build & Test

```bash
cargo build              # compile all features
cargo test               # run unit tests
cargo build --features zmq  # compile with ZMQ backend
```

## Architecture

- `src/trait_def.rs` — RuntimeBackend trait: process, initialize, save_state, get_spike_states, reset
- `src/models.rs` — Wire protocol: RuntimeSnapshot, RuntimeMessage enum (serde externally-tagged), SpikeBatch, SpikeEvent, TraceData, ConfigPayload
- `src/zmq_backend.rs` — ZMQ Subscriber backend (feature-gated zmq): 8-byte tick header + f32 readout
- `src/rust_backend.rs` — Pure-Rust native backend, no external deps, push-pull encoding
- `src/error.rs` — BackendError enum: CommunicationError, InitializationError, ModelIoError, ProcessingError, InvalidInputError
- `src/bin/corpus_ipc_server.rs` — HTTP microservice (axum + tokio): /initialize, /process, /save_state, /reset

## Tools

- `cargo build` / `cargo test` / `cargo clippy` — standard Rust toolchain
- `cargo fmt --check` — formatting check (CI enforced)
- GitHub Actions CI runs: fmt, clippy, build, test with all features

## Conventions

- Edition 2024, dual MIT (Massachusetts Institute of Technology) / Apache-2.0
- `#[must_use]` on constructors, `const fn` where possible
- Use `Ok(())` not `Ok(_)` for unit returns
- serde aliases for wire-protocol backward compatibility

## Constraints

- Do NOT rename literature citations in doc comments without verifying the actual paper title
- Do NOT change serde field names without adding `#[serde(alias = "...")]` for backward compatibility
- Avoid domain-specific naming (Nero, Brain, Spine) — prefer generic runtime/IPC terms
- Do NOT commit secrets, tokens, or API keys to the repository
- `docs/plans/` is gitignored — avoid tracking planning documents

## CI

```bash
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-features && cargo test --all-features
```
