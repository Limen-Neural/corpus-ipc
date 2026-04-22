# corpus-ipc

Inter-Process Communication (IPC) library for bridging Rust to external compute engines.

`corpus-ipc` is the schema and transport layer for hybrid Rust↔Julia workflows. It provides backend abstractions for local/native execution and optional ZeroMQ IPC, plus canonical wire message models used across services.

## Features

- `NeuralBackend` trait for backend-agnostic signal processing
- `RustBackend` reference backend (always available)
- `ZmqBrainBackend` backend via ZMQ SUB socket (feature `zmq`)
- Canonical protocol models:
  - `SpineMessage`
  - `SpikeBatch`, `SpikeEvent`
  - `EmbeddingBatch`
  - `GradientBatch`, `GradientUpdate`
  - `TraceBatch`, `TraceData`
  - `ConfigPayload`, `ConfigValue`, `BatchMetadata`
- `HybridFlowBackend` trait for message-oriented hybrid transports
- `NeroManifoldSnapshot` for parsed neuromodulator readout payloads

## Installation

```toml
[dependencies]
corpus-ipc = { git = "https://github.com/Limen-Neural/corpus-ipc" }

# Optional ZMQ backend support
# corpus-ipc = { git = "https://github.com/Limen-Neural/corpus-ipc", features = ["zmq"] }
```

## Quick Start

```rust
use corpus_ipc::{BackendFactory, BackendType, NeuralBackend};

let mut backend = BackendFactory::create(BackendType::Rust);
backend.initialize(None)?;

let inputs = [0.1_f32, -0.2, 0.3, 0.0];
let outputs = backend.process_signals(&inputs)?;

println!("{}", outputs.len());
# Ok::<(), corpus_ipc::BackendError>(())
```

## Protocol Ownership

`corpus-ipc` is the owner of serialized network schemas for hybrid flow messaging.

Use these re-exports directly from crate root:

```rust
use corpus_ipc::{SpineMessage, SpikeBatch, EmbeddingBatch};
```

## Crate Exports

- Backends and traits:
  - `NeuralBackend`, `HybridFlowBackend`
  - `BackendType`
  - `RustBackend`
  - `ZmqBrainBackend` (when `zmq` feature enabled)
- Models:
  - `SpineMessage` and all batch/config/trace/gradient payload structs
  - `NeroManifoldSnapshot`

## License

GPL-3.0-or-later
