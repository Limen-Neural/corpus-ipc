<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI workflow for automated validation (fmt, clippy, build, test) (#10).

### Changed

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
