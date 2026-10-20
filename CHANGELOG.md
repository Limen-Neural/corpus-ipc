<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Structured ZeroMQ `IpcMessage` transport behind the `zmq` feature (LIM-1278,
  consolidating LIM-1330). `ZmqMessagePublisher` binds and `ZmqMessageSubscriber`
  connects; each message is one compact canonical JSON `WireEnvelope` of
  `IpcMessage`. The publisher validates before `encode_canonical_ipc_message`.
  The subscriber decodes with `decode_ipc_message_json` and `Validate`.
  `ZmqTransportConfig` sets endpoint, send/receive high-water marks, linger, and
  reconnect limits before bind/connect. The reconnect cap must be at least the
  reconnect interval; libzmq ignores a smaller cap. The publisher and subscriber
  retain the ZeroMQ context until the socket is closed. `ZmqHybridFlowBackend` implements
  `HybridFlowBackend` on that pair: spikes and embeddings go out, gradients and
  eligibility traces come in. The other of those two inbound variants is held
  for its receiver; a second held sibling returns `CommunicationError` instead
  of being dropped. Any unrelated inbound variant is
  `BackendError::Protocol`. Identical outbound and inbound endpoints are
  rejected before either socket opens. A high-water mark of `0` is unlimited.
  Inbound frames are capped (`ZMQ_MAXMSGSIZE`, 512 MiB). A publisher rejects
  an encoded frame above that cap before send. Multipart input is drained up
  to eight extra frames. A message that fits in that drain returns a framing
  error and leaves the subscriber open; frames still remaining after the drain
  close the subscriber.
  Validation failures inside an envelope payload surface as `InvalidInput` on
  the hybrid adapter. A closed publisher reports `Closed` before payload
  validation. PUB/SUB stays
  best-effort and at-most-once (slow-joiner loss, high-water-mark loss, no
  acknowledgements), including for control and config messages. `ZmqIpcBackend`
  remains the separate binary `[i64 tick][N × f32 LE]` readout subscriber and
  does not implement `HybridFlowBackend`. Socket ownership stays
  mutex-serialized `Send` + `Sync` with no `unsafe impl Sync`.

- **v0.2 REST contract** for `corpus_ipc_server` (feature `server`), now on
  Axum 0.8 (LIM-1329). The router lives in `corpus_ipc::server::router` so it
  can be exercised in-process without binding a socket; the binary is a thin
  `main` over it. `docs/rest-api.md` documents the four endpoints
  (`POST /initialize`, `/process`, `/save_state`, `/reset`), the success and
  error envelopes, and the HTTP status classes, and ships in the crates.io
  package.

- Bounded ZMQ readout parser (`zmq_readout::parse_readout_packet`) shared by
  production recv and tests; golden LE fixtures + SHA-256 manifest under
  `test-vectors/zmq/` (LIM-1275, LIM-1328). Default decoded readout cap: 1024
  floats (`CORPUS_IPC_ZMQ_MAX_READOUT_FLOATS` override).

- Repository-only cargo-fuzz targets in `fuzz/` (LIM-1277):
  `fuzz_ipc_message_json` (compatibility decoder), `fuzz_zmq_readout_frame`
  (`parse_readout_packet` and `ZmqIpcBackend` ingest), and
  `fuzz_zmq_message_frame` (the structured subscriber's frame decoder). All
  three check no-panic. The two JSON targets also check `ProtocolLimits`
  enforcement and canonical round trips; the readout target checks frame
  bounds, bit-exact decoding, and that the backend cache changes only on
  accept.
  `scripts/fuzz-smoke.sh` runs a bounded campaign seeded in place from
  `test-vectors/`; the Linux-only `Fuzz smoke` workflow runs it for 60 s per
  target. `tests/fuzz_regressions.rs` replays the same invariants and any
  committed regression seeds on stable. `fuzz/` is outside the package
  `include` list, so the published crate gains no dependency or binary.
- `ZmqMessageSubscriber::decode_frame_for_tests` (`#[doc(hidden)]`, feature
  `zmq`): the single-frame decoder `try_recv` runs, without a socket. The
  empty-frame check moved into that decoder; `try_recv` behavior is unchanged.

### Changed

- Direct dependency `thiserror` is 2.x (LIM-1451). `BackendError`,
  `ProtocolError`, `CompatibilityError`, `EnvelopeError`, and
  `CanonicalEncodeError` keep the same Display text, `Error::source`
  results, and variant matches. Compatible lockfile updates are included.
  The Axum crate stays 0.8. The crate version stays 0.1.0.

- `BackendError` gains `Protocol(ProtocolError)` so a hybrid-flow receiver can
  report an unexpected `IpcMessage` variant without overloading transport or
  validation errors. Exhaustive matches on `BackendError` need a new arm. Wire
  version stays 1. The REST mapper still treats this arm as a backend failure
  (HTTP 502); the router does not expose hybrid-flow sockets.

- crates.io metadata now describes the spike and neuromodulator wire protocol
  (description, keywords `snn` / `neuromorphic` / `spiking` / `ipc` / `zmq`,
  categories `science::neuroscience`, `network-programming`, `encoding`).
  The README intro and crate-level docs use the same sentence. No API change
  and no version bump (GitHub #53, LIM-1524).

- **0.1 REST → v0.2 REST migration (not wire-compatible).** The 0.1 server
  answered every application-level result with HTTP 200 and
  `{"ok": false, "message"}` on backend errors. v0.2 uses real status
  classes: 400 malformed JSON, 415 missing/wrong content type, 422
  schema/input validation failures, 409 use before `/initialize` or after
  `/reset`, 502 backend failures, 500 internal failures. Error bodies are
  `{"ok": false, "error": {"code", "message"}}`. Success bodies gain
  `"ok": true` (`/process` returns `{"ok": true, "output": [...]}`);
  request bodies are unchanged. Axum/Tokio stay absent from default
  consumers (`server` remains opt-in; `tower` is a dev-dependency for router
  tests only).

- `ZmqIpcBackend` rejects malformed or over-limit readout frames with
  `BackendError` instead of logging and retaining cache; `EAGAIN` still returns
  cached readout without error. Compatible with 0.1.x patch expectations for
  supported frame sizes (≤ cap, aligned); frames above the cap are newly
  rejected.

- Canonical wire-version-1 encoder `encode_canonical_ipc_message` and dedicated
  `CanonicalEncodeError` (LIM-1326). Produces deterministic, sorted-key bytes
  for the project wire profile (documented in `docs/wire-encoding.md`), reusing
  the `WireEnvelope` model and `Validate` policy. Non-finite floats are rejected
  before serialization with a typed error instead of a silent JSON `null`.
  `ConfigValue` stays Float-first: `Integer(42)` round-trips as `Float(42.0)`
  and integers above `2^24` can lose precision or identity on an `f32`
  round-trip (documented, not redesigned). Canonical `f32` bytes use `f64`
  promotion so they stay stable under serde_json `arbitrary_precision`
  unification. The crates.io package includes `docs/wire-encoding.md`.
  `ConfigValue` JSON decimals (`1.0`, `0.1`) still decode as `Float` when
  serde_json `arbitrary_precision` is unified.

## [0.1.0]

Initial public release of corpus-ipc.

### Added

- Typed `ValidationError` / `ValidationKind` and `ProtocolLimits` for wire
  payload validation (LIM-1231). `validate()` is implemented for every public
  batch and message payload; deserialization reuses the same checks and
  rejects oversize collections as early as serde permits.
- Wire-schema compatibility envelope (RM-1333): `WireCompatibility` is the
  public source of truth for current (`1`) and minimum supported (`1`) wire
  versions. `decode_ipc_message_json` classifies incoming envelopes as
  supported, too old, or too new and returns typed errors **before** the
  payload is used. Unversioned pre-release `IpcMessage` JSON remains accepted
  as legacy wire version 1. Unknown struct fields are ignored (forward
  compatible); unknown `IpcMessage` variants never deserialize as a default.
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
  `homepage`, `documentation`, docs.rs config, and an explicit package
  `include` list (#33, RM-1202).
- Deprecated compatibility aliases after the #20 IPC rename (RM-334, #13, LIM-169):
  - `RuntimeBackend` → `IpcBackend`
  - `ZmqRuntimeBackend` → `ZmqIpcBackend`
  - `BackendType::ZmqRuntime` → `BackendType::ZmqIpc`
  Both current and deprecated ZMQ selectors construct `ZmqIpcBackend`.
  Compile-time tests assert that `RustBackend` and `ZmqIpcBackend` implement `IpcBackend`.

### Changed

- **Breaking:** `validate()` now returns `Result<(), ValidationError>` instead
  of `Result<(), String>`. Deserialization of public wire payloads enforces
  `ProtocolLimits::DEFAULT`, so previously accepted oversize or non-finite
  payloads are rejected.
- Point live crate metadata and docs at `Limen-Neural/corpus-ipc` after the
  ownership transfer (`rmems/corpus-ipc` redirects here). Document that the
  GitHub wiki is enabled. (RM-309, #24)
- `serde_json` is a library dependency (feature `raw_value`) so the
  envelope decoder can inspect `wire_version` before deserializing a payload.
  This reverses the earlier pre-release move of `serde_json` to a test-only
  (dev) dependency.
- GitHub Actions Codecov workflow (`cargo llvm-cov` LCOV upload with org
  `CODECOV_TOKEN`, slug `Limen-Neural/corpus-ipc`) (#34, RM-1203).
- Feature-gated the HTTP service stack behind `server` so the default crate no
  longer depends on Axum/Tokio/Tower. `corpus_ipc_server` now requires
  `--features server` (`required-features = ["server"]`). Tokio is limited to
  `macros`, `net`, and `rt-multi-thread`. `serde_json` stays a library
  dependency (feature `raw_value`) for the wire-compatibility envelope
  decoder. CI checks default, `zmq`, `server`, and `--all-features` graphs
  (#29, RM-1142).
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

- `ZmqIpcBackend` no longer uses an unsound `unsafe impl Sync` on the wrapped
  libzmq socket (LIM-1232). `zmq` 0.10's `Socket` is `Send` + `!Sync`; the SUB
  socket is now stored in a `Mutex` so the public backend can stay `Send` +
  `Sync` (as `IpcBackend` requires) without claiming concurrent `&Socket`
  access is safe. Create/connect run on the initializing thread before the
  socket is published into that mutex; recv and `reset` close take `&mut self`
  on the exclusive owner behind the lock. Compile-time trait assertions lock
  the intended surface.
- Various minor clippy lints and formatting to enable strict CI enforcement (#10).
- Markdown lint in boundary plan (Codacy "spaces inside code span").

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
- adding a new `IpcMessage` variant that old readers will reject with
  `EnvelopeError::Payload` rather than `CompatibilityError::TooNew` (this is
  a crate API addition; old readers already fail closed on unknown
  variants — bump `CURRENT` if new producers must be distinguished from
  old ones at the envelope layer)

**Raise `MIN_SUPPORTED` when this crate drops decode of an older wire
version.** Keep at least one committed fixture for every still-supported
version, including the unversioned-as-v1 encoding while
  `LEGACY_UNVERSIONED` remains inside the window.

