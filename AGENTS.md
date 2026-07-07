# corpus-ipc

Rust Inter-Process Communication (IPC) library for bridging to external compute engines via ZeroMQ (ZMQ).
Part of the Limen-Neural neuromorphic computing stack.

Last updated: 2026-07-07

## Build & Test

```bash
cargo build              # compile all features
cargo test               # run unit tests
cargo build --features zmq  # compile with ZMQ backend
```

## Architecture

- `src/trait_def.rs` — RuntimeBackend trait (core abstraction): process, initialize, save_state, get_spike_states, reset
- `src/models.rs` — wire protocol types: RuntimeSnapshot (88-byte IPC packet), RuntimeMessage (serde enum), SpikeBatch, SpikeEvent, TraceData, ConfigPayload
- `src/zmq_backend.rs` — ZMQ Subscriber (SUB) backend (feature-gated `zmq`): reads 8-byte tick header + f32 readout floats
- `src/rust_backend.rs` — pure-Rust native backend (no external dependencies)
- `src/error.rs` — BackendError types (CommunicationError, InitializationError, etc.)
- `src/bin/corpus_ipc_server.rs` — HTTP server binary (axum + tokio)

## Conventions

- Edition 2024, dual MIT (Massachusetts Institute of Technology) / Apache-2.0 license
- `#[must_use]` on constructors, `const fn` where possible
- Use `Ok(())` not `Ok(_)` for unit returns
- serde aliases for wire-protocol backward compatibility
- `docs/plans/` is gitignored — do not track planning documents

## What NOT to Do

- Do NOT rename literature citations in doc comments (e.g. Schultz 1998 paper title must stay "dopamine neurons")
- Do NOT change serde field names without adding `#[serde(alias = "...")]` for backward compatibility
- Do NOT add domain-specific naming (Nero, Brain, Spine) — use generic runtime/IPC terms
- Do NOT commit secrets, tokens, or API keys

## CI

```bash
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo build --all-features && cargo test --all-features
```
