<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Wire-schema compatibility envelope (RM-1333): `WireCompatibility` is the
  public source of truth for current (`1`) and minimum supported (`1`) wire
  versions. `decode_ipc_message_json` classifies incoming envelopes as
  supported, too old, or too new and returns typed errors **before** the
  payload is used. Unversioned 0.1.0 `IpcMessage` JSON remains accepted as
  legacy wire version 1. Unknown struct fields are ignored (forward
  compatible); unknown `IpcMessage` variants never deserialize as a default.

### Changed

- `serde_json` is a library dependency again (feature `raw_value`) so the
  envelope decoder can inspect `wire_version` before deserializing a payload.
  This reverses the 0.1.0 move of `serde_json` to a test-only (dev)
  dependency.
- GitHub Actions Codecov workflow (`cargo llvm-cov` LCOV upload with org
  `CODECOV_TOKEN`, slug `Limen-Neural/corpus-ipc`) (#34, RM-1203).

### Wire version bumps

Crate semver and wire version are independent. Change
`WireCompatibility::CURRENT` / `MIN_SUPPORTED` (in `src/compatibility.rs`)
when the **on-wire** schema changes, not merely because the crate version
changed.

**Bump `CURRENT` (and treat it as a breaking wire change) when:**

- renaming, removing, or changing the type of a serialized field
- renaming, removing, or adding an `IpcMessage` variant that existing
  decoders must understand
- adding a required field with no default the older encoding can omit
- tightening unknown-field policy (for example `deny_unknown_fields`)
- changing tagged-enum identity (`Spikes`, `Ping`, …) or envelope keys
  (`wire_version`, `payload`)

If the new schema cannot still decode the previous payload, also raise
`MIN_SUPPORTED` to that new version (or keep a version-specific decoder
for the old one). Bumping only `CURRENT` leaves older versions inside
the accepted range.

**Do not bump `CURRENT` when:**

- only the Rust API, docs, or crate semver change
- adding an *optional* field that older readers can ignore under the
  unknown-field rule
- adding a new `IpcMessage` variant that old readers will reject (that is
  a crate API addition; old readers already fail closed on unknown
  variants — bump `CURRENT` if new producers must be distinguished from
  old ones at the envelope layer)

**Raise `MIN_SUPPORTED` when this crate drops decode of an older wire
version.** Keep at least one committed fixture for every still-supported
version, including the unversioned-as-v1 encoding while
`LEGACY_UNVERSIONED` remains inside the window.

## [0.1.0]

First crates.io publish (RM-1141, #28).

### Added

- `StimulusBatch` typed wire type and `IpcMessage::Stimuli` / `IpcMessage::Neuromodulators`
  variants for canonical, domain-neutral runtime stimulus and neuromodulator
  ingress (RM-1140, #27). Replaces the need for downstream services
  (`thalamic-relay`, `brainstem-daemon`) to use bespoke UDP JSON or private
  packet structs. Stimulus channel width is not fixed by this crate, and an
  optional `valid_mask` lets a channel be marked invalid/missing for a tick
  instead of silently reading as `0.0`.
- GitHub Actions CI workflow for automated validation (fmt, clippy, build, test) (#10).
- OS matrix for Build & Test on `ubuntu-latest`, `macos-latest`, and
  `windows-latest` (`fail-fast: false`). ZeroMQ / `--all-features` jobs stay
  Linux-only (#32, RM-1201).
- crates.io packaging metadata: `authors`, `rust-version` 1.98.1, `readme`,
  `homepage`, `documentation`, docs.rs config, and a tighter `exclude` list
  (#33, RM-1202).
- Deprecated compatibility aliases after the #20 IPC rename (RM-334, #13, LIM-169):
  - `RuntimeBackend` → `IpcBackend`
  - `ZmqRuntimeBackend` → `ZmqIpcBackend`
  - `BackendType::ZmqRuntime` → `BackendType::ZmqIpc`
  Both current and deprecated ZMQ selectors construct `ZmqIpcBackend`.
  Compile-time tests assert that `RustBackend` and `ZmqIpcBackend` implement `IpcBackend`.

### Changed

- Feature-gated the HTTP service stack behind `server` so the default crate no
  longer depends on Axum/Tokio/Tower. `corpus_ipc_server` now requires
  `--features server` (`required-features = ["server"]`). Tokio is limited to
  `macros`, `net`, and `rt-multi-thread`. `serde_json` moved to a
  dev-dependency for wire-format tests. CI checks default, `zmq`, `server`,
  and `--all-features` graphs (#29, RM-1142).
- `corpus_ipc_server` defaults to `127.0.0.1:8080`. Set `CORPUS_IPC_BIND` for
  another address; remote binds need an external access-control boundary.
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
[0.1.0]: https://github.com/Limen-Neural/corpus-ipc/releases/tag/v0.1.0
