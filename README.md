<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc

Inter-Process Communication (IPC) library for bridging Rust to external compute engines.

`corpus-ipc` is the schema and transport layer for hybrid Rust↔Julia workflows. It provides backend abstractions for local/native execution and optional ZeroMQ IPC, plus canonical wire message models used across services.

## Features

- `RuntimeBackend` trait for backend-agnostic signal processing (was ComputeBackend)
- `RustBackend` reference backend (always available)
- `ZmqRuntimeBackend` backend via ZMQ SUB socket (feature `zmq`)
- Canonical protocol models:
  - `RuntimeMessage`
  - `SpikeBatch`, `SpikeEvent`
  - `EmbeddingBatch`
  - `GradientBatch`, `GradientUpdate`
  - `TraceBatch`, `TraceData`
  - `ConfigPayload`, `ConfigValue`, `BatchMetadata`
- `HybridFlowBackend` trait for message-oriented hybrid transports
- `RuntimeSnapshot` for parsed runtime readout payloads

## Installation

```toml
[dependencies]
corpus-ipc = { git = "https://github.com/Limen-Compute/corpus-ipc" }

# Optional ZMQ backend support
# corpus-ipc = { git = "https://github.com/Limen-Compute/corpus-ipc", features = ["zmq"] }
```

## Quick Start

```rust
use corpus_ipc::{BackendType, RuntimeBackend};
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
use corpus_ipc::{RuntimeMessage, SpikeBatch, EmbeddingBatch};
```

## Crate Exports

- Backends and traits:
  - `RuntimeBackend`, `HybridFlowBackend`
  - `BackendType`
  - `RustBackend`
  - `ZmqRuntimeBackend` (when `zmq` feature enabled)
- Models:
  - `RuntimeMessage` and all batch/config/trace/gradient payload structs
  - `RuntimeSnapshot`

## License

Dual-licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE-2.0 or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License (LICENSE-MIT or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
