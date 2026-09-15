# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build                                          # default features (no zmq, no HTTP server)
cargo check --no-default-features
cargo check --features zmq
cargo check --features server                         # also typechecks corpus_ipc_server
cargo test                                            # default features
cargo test <test_name>                                # single test, e.g. `cargo test stimulus_batch_round_trips`
cargo test --all-features                             # includes zmq backend + its tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings             # CI uses --all-features too; run that when touching zmq_backend.rs
cargo doc --no-deps
```

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `cargo check` for default / `zmq` / `server`, a `cargo tree --no-default-features` assertion that Axum/Tokio/Tower/ZMQ stay out of the core graph, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --all-features`, and `cargo test --all-features` — match that locally before pushing.

**The `zmq` feature needs a working C++ compiler** (`zmq-sys` builds vendored ZeroMQ from C++ source, it does not link system libzmq). If `c++`/`cc` aren't available, `--all-features` builds fail with a `cc-rs` error unrelated to this crate's code — this is an environment gap, not a regression. If clang can't find libstdc++ headers, force gcc/g++:

```bash
CC=gcc CXX=g++ cargo build --all-features
```

Run the REST service (binary `corpus_ipc_server`, auto-discovered from `src/bin/`):

```bash
CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features server --bin corpus_ipc_server
# ZMQ backend selectable via CORPUS_IPC_BACKEND_TYPE=zmq:
CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features server,zmq --bin corpus_ipc_server
```

Backend selection and endpoints are via env vars: `CORPUS_IPC_BACKEND_TYPE` (`zmq` or default `Rust`), `CORPUS_IPC_BIND` (listen address, default `0.0.0.0:8080`), `CORPUS_IPC_ZMQ_READOUT_IPC` (ZMQ SUB endpoint, default `ipc:///tmp/corpus_ipc_readout.ipc`). Exercise the service with `POST /initialize`, `POST /process {"inputs":[...]}`, `POST /save_state {"model_path":"..."}`, `POST /reset`.

## Architecture

`corpus-ipc` is the schema/transport layer bridging Rust to an external compute engine (a Julia SNN runtime). It has two largely independent halves that are easy to conflate:

1. **`IpcBackend` trait + implementations** (`trait_def.rs`, `rust_backend.rs`, `zmq_backend.rs`) — a synchronous `process_batch(&[f32]) -> Vec<f32>` contract, selected at runtime via `BackendFactory::create(BackendType)`. `RustBackend` is a stateless push-pull-encoding stub (positive input → even channel, negative magnitude → odd channel); `ZmqIpcBackend` (feature `zmq`) is a **binary** readout subscriber over a ZMQ SUB socket — 8-byte tick header + N×f32 LE readout, not a serde/JSON message. This layer has no concept of `IpcMessage`.
2. **Wire message models** (`models.rs`) — `IpcMessage`, a serde-tagged enum of typed payloads (`SpikeBatch`, `EmbeddingBatch`, `StimulusBatch`, `NeuromodulatorSnapshot`, `GradientBatch`, `TraceBatch`, config/control messages) used for structured hybrid-flow messaging. `HybridFlowBackend` is the optional trait for backends that exchange these structured messages (currently no in-repo implementation — `IpcBackend` is the only trait every backend must satisfy).

`src/bin/corpus_ipc_server.rs` is a thin axum REST wrapper around `IpcBackend` only (not the `IpcMessage` models) — one backend instance behind a `Mutex`, one route per trait method.

### Downstream architecture this crate serves

```text
thalamic-relay -> corpus-ipc -> brainstem-daemon
 sensory/safety    wire/schema      SNN runtime
```

`corpus-ipc` owns the shared schema so `thalamic-relay` and `brainstem-daemon` don't need bespoke UDP JSON or private packet structs. Keep this crate's models domain-neutral: no telemetry/NVML concepts, no hardware policy, no SNN stepping, no training logic, and no network-specific fixed widths (e.g. don't bake a specific channel count into a wire type) — those belong in the downstream repos.

### Naming: `Ipc*` is current, `Runtime*` is deprecated

PR #20 renamed the public API for generic IPC terminology: `RuntimeBackend` → `IpcBackend`, `RuntimeMessage` → `IpcMessage`, `RuntimeSnapshot` → `NeuromodulatorSnapshot`, `ZmqRuntimeBackend` → `ZmqIpcBackend`, `BackendType::ZmqRuntime` → `BackendType::ZmqIpc`. The old names still exist as `#[deprecated]` aliases for downstream migration — don't remove them casually, and don't reintroduce `Runtime*` naming in new code.

### `SpikeBatch`/`TraceBatch` name collision with `SynapticDistill.jl`

This crate's `SpikeBatch`/`TraceBatch` are **IPC wire payloads** (session/batch id + typed rows), deliberately keeping their Rust names for serde/import stability even though [`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl) has same-named but structurally different **training-side** types. Use the `IpcSpikeBatch`/`IpcTraceBatch` aliases when writing code that sits next to the training types to avoid ambiguity. Never merge the two definitions.

### Wire-type invariant pattern: public fields + opt-in `validate()`

Wire structs (`StimulusBatch`, `NeuromodulatorSnapshot`) keep all fields `pub` (matching `SpikeBatch`/`TraceBatch`) rather than hiding them behind a validating constructor, since arbitrary deserialization can always bypass a constructor anyway. Documented invariants (e.g. `StimulusBatch.valid_mask` length must match `values`; `NeuromodulatorSnapshot`'s per-field ranges) are instead checked by a `validate() -> Result<(), String>` method that callers invoke explicitly after building or deserializing. Follow this pattern for new wire types with invariants rather than inventing a different validation style.

### Serialization compatibility tests are load-bearing

`models.rs`'s test module locks `IpcMessage` variant names and struct field names via `serde_json::to_value` snapshots (e.g. `spike_batch_json_keys_stay_stable`, `ipc_message_envelopes_keep_variant_names`). These aren't incidental — they're the contract with other repos deserializing this wire format. Add an equivalent test for any new `IpcMessage` variant or wire struct, and don't change existing field/variant names without treating it as a breaking wire change.
