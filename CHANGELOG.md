<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `StimulusBatch` typed wire type and `IpcMessage::Stimuli` / `IpcMessage::Neuromodulators`
  variants for canonical, domain-neutral runtime stimulus and neuromodulator
  ingress (RM-1140, #27). Replaces the need for downstream services
  (`thalamic-relay`, `brainstem-daemon`) to use bespoke UDP JSON or private
  packet structs. Stimulus channel width is not fixed by this crate, and an
  optional `valid_mask` lets a channel be marked invalid/missing for a tick
  instead of silently reading as `0.0`.
- GitHub Actions CI workflow for automated validation (fmt, clippy, build, test) (#10).
- Deprecated compatibility aliases after the #20 IPC rename (RM-334, #13, LIM-169):
  - `RuntimeBackend` → `IpcBackend`
  - `ZmqRuntimeBackend` → `ZmqIpcBackend`
  - `BackendType::ZmqRuntime` → `BackendType::ZmqIpc`
  Both current and deprecated ZMQ selectors construct `ZmqIpcBackend`.
  Compile-time tests assert that `RustBackend` and `ZmqIpcBackend` implement `IpcBackend`.

### Changed

- Documented the rationale for keeping the `SpikeBatch` / `TraceBatch` Rust
  names: serde identifiers and existing imports stay stable. No wire-format
  change (RM-324, #7).
- Switched license from GPL-3.0-or-later to dual MIT/Apache-2.0 for broader adoptability as core IPC infrastructure (#11).
- **Breaking rename**: Generalized IPC terminology across the public API (#8):
  - `RuntimeBackend` → `IpcBackend`
  - `RuntimeMessage` → `IpcMessage`
  - `RuntimeSnapshot` → `NeuromodulatorSnapshot`
  - `ZmqRuntimeBackend` → `ZmqIpcBackend`
  - `BackendType::ZmqRuntime` → `BackendType::ZmqIpc`
  - Server binary target renamed to `corpus_ipc_server`
  - Environment variables renamed: `CORPUS_IPC_BACKEND_TYPE`, `CORPUS_IPC_BIND`, `CORPUS_IPC_ZMQ_READOUT_IPC`
  - Downstream users must update all renamed types, enum variants, and environment contract usage.
- Cleaned legacy terminology in zmq logs for neutral boundary (as part of combined #4 work).

### Fixed

- Various minor clippy lints and formatting to enable strict CI enforcement (#10).
- Markdown lint in boundary plan (Codacy "spaces inside code span").

[Unreleased]: https://github.com/Limen-Neural/corpus-ipc/compare/main...HEAD
