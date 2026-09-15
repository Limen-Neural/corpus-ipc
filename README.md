<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc

Inter-Process Communication (IPC) library for bridging Rust to external compute engines.

`corpus-ipc` is the schema and transport layer for cross-process compute workflows. It provides backend abstractions for local/native execution and optional ZeroMQ IPC, plus canonical wire message models used across services.

## Features

- `IpcBackend` trait for backend-agnostic signal processing (deprecated alias: `RuntimeBackend`)
- `RustBackend` reference backend (always available)
- `ZmqIpcBackend` backend via ZMQ SUB socket (feature `zmq`; deprecated alias: `ZmqRuntimeBackend`)
- Optional `corpus_ipc_server` REST binary (feature `server`; not compiled for library consumers)
- Canonical protocol models:
  - `IpcMessage`
  - `SpikeBatch`, `SpikeEvent` (IPC wire types; alias `IpcSpikeBatch`)
  - `EmbeddingBatch`
  - `StimulusBatch` for typed, variable-width runtime stimulus ingress
  - `GradientBatch`, `GradientUpdate`
  - `TraceBatch`, `TraceData` (IPC wire types; alias `IpcTraceBatch`)
  - `ConfigPayload`, `ConfigValue`, `BatchMetadata`
- `HybridFlowBackend` trait for message-oriented hybrid transports
- `NeuromodulatorSnapshot` for typed neuromodulator ingress/readout payloads

## Feature flags

The default dependency graph is wire models plus `RustBackend`. HTTP and ZeroMQ
are opt-in so a consumer of `IpcMessage` / `StimulusBatch` does not compile or
link those stacks.

| Feature | Default | What it enables |
| --- | --- | --- |
| *(none)* | yes | Wire models (`IpcMessage`, batches, snapshots) and `RustBackend` |
| `zmq` | no | `ZmqIpcBackend` (vendored libzmq via `zmq-sys`; needs a C++ compiler). `Send` + `Sync` via mutex-serialized SUB socket ownership; libzmq sockets themselves are `Send` + `!Sync`. |
| `server` | no | `corpus_ipc_server` Axum REST binary (`axum` + minimized `tokio`) |

`tower` is not a direct crate dependency. Axum 0.7 depends on it unconditionally,
so it appears only when the optional `server` feature enables Axum. `serde_json`
is a **dev-dependency** for serialization-contract tests in `models.rs`; the
server binary uses Axum's `Json` extractor (the `json` feature), which depends
on `serde_json` internally.

### ZMQ backend concurrency

`ZmqIpcBackend` is `Send` + `Sync` because `IpcBackend` requires both. libzmq
sockets are **not** safe to share (`zmq(7)` *Thread safety*; `zmq` 0.10's
`Socket` is `Send` + `!Sync`). This crate does not `unsafe impl Sync` on the
socket. The SUB socket is stored in a `Mutex` and every recv/connect/close
takes that lock. Moving the backend to another thread is supported. `initialize`,
`process_batch`, and `reset` still take `&mut self`; concurrent process calls
need an outer lock, which is how `corpus_ipc_server` uses
`Arc<Mutex<Box<dyn IpcBackend>>>`.

## Installation

```toml
[dependencies]
corpus-ipc = { git = "https://github.com/Limen-Neural/corpus-ipc" }

# Optional ZMQ backend support
# corpus-ipc = { git = "https://github.com/Limen-Neural/corpus-ipc", features = ["zmq"] }
```

Run the REST service from this repo (not pulled in by a library dependency):

```bash
# RustBackend only
CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features server --bin corpus_ipc_server

# With ZMQ backend selectable via CORPUS_IPC_BACKEND_TYPE=zmq
CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features server,zmq --bin corpus_ipc_server
```

## Quick Start

```rust
use corpus_ipc::{BackendType, IpcBackend};
use corpus_ipc::trait_def::BackendFactory;

let mut backend = BackendFactory::create(BackendType::Rust);
backend.initialize(None)?;

let inputs = [0.1_f32, -0.2, 0.3, 0.0];
let outputs = backend.process_batch(&inputs)?;

println!("{}", outputs.len());
# Ok::<(), corpus_ipc::BackendError>(())
```

## Protocol Ownership

`corpus-ipc` is the owner of serialized network schemas for hybrid flow messaging.

Use these re-exports directly from crate root:

```rust
use corpus_ipc::{IpcMessage, SpikeBatch, EmbeddingBatch};
```

### `SpikeBatch` / `TraceBatch` are IPC transport types

These names are **wire payloads**, not the training-side types in
[`SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)
(`src/types.jl`).

| Crate | Domain | `SpikeBatch` | `TraceBatch` |
| --- | --- | --- | --- |
| `corpus-ipc` | IPC transport | session/batch id + `SpikeEvent` list | session/batch id + `TraceData` rows |
| `SynapticDistill.jl` | SNN training | spike trains + optional `times` / `targets` | unstructured e-prop `traces` |

The Rust type names stay `SpikeBatch` / `TraceBatch` so serde identifiers
(`IpcMessage::Spikes`, `IpcMessage::EligibilityTraces`) and existing imports
remain compatible. When the collision would be confusing, use the aliases
`IpcSpikeBatch` and `IpcTraceBatch` — they are the same types and the same
wire format.

### Stimulus and neuromodulator ingress

`corpus-ipc` is the schema owner for the stimulus/neuromodulator leg of the
sensory pipeline. The intended flow is:

```text
thalamic-relay -> corpus-ipc -> brainstem-daemon
 sensory/safety    wire/schema      SNN runtime
```

`thalamic-relay` and `brainstem-daemon` encode/decode through
`IpcMessage::Stimuli(StimulusBatch)` and
`IpcMessage::Neuromodulators(NeuromodulatorSnapshot)` instead of bespoke UDP
JSON or a private packet struct. `StimulusBatch` is domain-neutral: its
`values` width is not fixed by this crate (do not encode any one network's
axon/channel count here), and an optional `valid_mask` lets a channel be
marked invalid/missing for a tick instead of silently reading as `0.0`. See
the `StimulusBatch` doc comments for the exact semantics.

## Crate Exports

- Backends and traits:
  - `IpcBackend`, `HybridFlowBackend`
  - `BackendType` (`Rust`, `ZmqIpc`; deprecated `ZmqRuntime` still selects ZMQ)
  - `RustBackend`
  - `ZmqIpcBackend` (when `zmq` feature enabled; mutex-serialized socket ownership)
  - Deprecated compatibility aliases: `RuntimeBackend`, `ZmqRuntimeBackend`
- Models:
  - `IpcMessage` and all batch/config/trace/gradient payload structs
  - `IpcSpikeBatch` / `IpcTraceBatch` aliases for the IPC wire batches
  - `StimulusBatch`, `NeuromodulatorSnapshot`

## License

Dual-licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE-2.0 or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License (LICENSE-MIT or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
