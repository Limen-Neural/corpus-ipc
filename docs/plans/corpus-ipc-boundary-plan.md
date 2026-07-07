<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc runtime boundary cleanup plan

- Linear issue: LIM-10
- GitHub tracking: <https://github.com/Limen-Neural/corpus-ipc/issues/3>
- Status: implementation in progress

## Purpose

Define the target boundary for `corpus-ipc` as a reusable runtime/IPC crate that:

- Owns protocol models and transport-level integration points.
- Exposes generic runtime/IPC abstractions that are not tied to a specific product domain.
- Removes legacy trader/NERO-specific wording from the planning target.

## Owns

`corpus-ipc` should own:

- Wire protocol types and serde schemas for cross-process messaging.
- Transport-neutral backend traits used to process/relay messages.
- Optional transport adapters (e.g. ZMQ), behind feature gates.
- Error types for runtime + serialization + transport boundary handling.
- Crate-root re-exports for stable IPC-facing types.

## Does not own

`corpus-ipc` should not own:

- Trading strategy logic, order lifecycle, exchange connectivity, or portfolio rules.
- Supervisor/app orchestration responsibilities (process lifecycle, deployment topology, product policy).
- Product-domain branding vocabulary in core trait/type names.
- Cross-repo business semantics that are not required for generic IPC/runtime boundaries.

## Source audit performed (this revision)

To ensure the plan is grounded in the actual crate surface, this revision explicitly reviewed the current Rust sources:

- `src/lib.rs`
- `src/trait_def.rs`
- `src/models.rs`
- `src/zmq_backend.rs`
- `src/rust_backend.rs`
- `src/bin/corpus_ipc_server.rs`
- `src/error.rs`
- `Cargo.toml`
- `README.md`

Observation:

- No `TraderBackend` type, trait, or public API symbol remains (the reference in the error module doc comment has been updated to the generic `BackendConnector` per #4).
- The source audit list has been expanded to include `src/error.rs`.
- Legacy/domain wording has been cleaned from public traits, exported models, backend names, env vars, and service binary naming (addressed in #4 + #5).

## Legacy wording inventory

The following names previously leaked legacy naming into the generic IPC surface (addressed by #4 + #5):

1. ~~`NeroManifoldSnapshot`~~ → `Snapshot` (model type and re-export).
2. ~~`ZmqBrainBackend`~~ → `ZmqBackend` (backend naming); `BackendType::ZmqBrain` → `Zmq`.
3. ~~`SpineMessage`~~ → `Message` (biological/product-coded framing replaced with neutral IPC envelope term).
4. ~~`process_signals`~~ → `process` (in `BackendConnector` trait method).

**Note on service/entrypoint rename (prior PR):** The service binary, env var names, and default endpoint were hard-renamed with no legacy aliases. Items above were completed in the #4 + #5 pass. Downstream consumers must migrate.

Additional cleanup targets discovered in metadata/docs (deferred to later stage; not changed in this PR's service/entrypoint rename pass):

- README references to neural/hybrid-specific naming in public API lists.

## Public API target (planning)

Target API shape for a generic runtime/IPC core:

- Keep transport-neutral trait boundary, but rename domain-coded trait/method identifiers.
- Keep canonical protocol structs, but migrate to neutral message naming conventions.
- Keep optional feature-gated transport adapters, but use neutral transport naming.
- Keep crate-root exports, but avoid product-coded type names in exported surface.

Representative naming direction (non-binding planning examples):

- `BackendConnector` (was NeuralBackend) -> `RuntimeBackend`
- `process_signals` -> `process_batch` (or equivalent neutral verb)
- `SpineMessage` -> `RuntimeMessage` / `Envelope`
- `ZmqBrainBackend` -> `ZmqRuntimeBackend`
- `NeroManifoldSnapshot` -> `RuntimeSnapshot` (or domain-neutral payload name)

## Dependency boundary target

`corpus-ipc` should remain focused and layered:

- Core layer: `serde`, shared model definitions, trait definitions, core errors.
- Optional transport layer: `zmq` feature for ZeroMQ adapter implementation.
- Utility/runtime support: async/server dependencies only if they directly support IPC runtime boundary.

Boundary rule:

- Product-app dependencies and business-domain crates must not be required by core IPC types.

## Migration risks

1. **Breaking API changes**: trait/type renames will require coordinated downstream updates.
2. **Serialization compatibility**: renaming message/type fields could impact existing consumers if wire schema changes.
3. **Documentation drift**: README and usage examples can become stale during staged renaming.
4. **Feature split ambiguity**: unclear separation between pure schema crate and runtime transport crate may create churn.
5. **Service binary/env/default hard rename**: This PR executed a hard cutover for the server binary target name, env var keys, and ZMQ default path (no legacy fallbacks in code). Deployment scripts, Dockerfiles, systemd units, CI, and external publishers (e.g. using old default IPC path or prior env var names) must be updated concurrently. Cargo bin target implicitly changes from old filename-based name to `corpus_ipc_server`.

## Open questions

1. Should wire-format stability be preserved via serde aliases during renaming, or is a versioned schema bump acceptable?
2. Should runtime trait abstractions and wire models remain in one crate, or be split into `core-schema` + `runtime-adapters` crates?
3. Which binaries (if any) remain in this repository versus moving into app/supervisor repos?
4. What is the deprecation window for legacy names before full removal?

## Proposed migration sequence (planning only)

1. Approve boundary + naming matrix in issue #3.
2. Finalize neutral terminology map (old -> new) and compatibility policy, including **all legacy env vars and default endpoint strings** (now migrated to `CORPUS_IPC_*`).
3. Stage non-breaking docs and type alias/deprecation pass.
4. Execute major rename pass with release notes and downstream coordination.
5. Remove deprecated legacy names after agreed window.
