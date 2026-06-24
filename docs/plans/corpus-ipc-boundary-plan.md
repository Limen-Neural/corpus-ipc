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
- `src/bin/spikenaut_server.rs`
- `Cargo.toml`
- `README.md`

Observation:

- **No direct `TraderBackend` symbol is present** in the current source tree.
- Legacy/domain wording still appears through neural/brain/spine/NERO/spikenaut terminology in public traits, exported models, backend names, env vars, and service binary naming.

## Legacy wording inventory

The following names currently leak legacy naming into the generic IPC surface (verified from the source files above):

1. `NeroManifoldSnapshot` model type and re-export.
2. `ZmqBrainBackend` backend naming (`Brain` is product/domain-coded wording).
3. `SpineMessage` naming (biological/product-coded framing instead of neutral IPC envelope terms).
4. `process_signals` in `NeuralBackend` trait method (domain-coded behavior wording).
5. `spikenaut_server` binary target naming (legacy app-specific wording).
6. `SPIKENAUT_BACKEND_TYPE` environment variable in `src/bin/spikenaut_server.rs`.
7. `SPIKENAUT_BIND` environment variable in `src/bin/spikenaut_server.rs`.
8. `SPIKENAUT_ZMQ_READOUT_IPC` environment variable in `src/zmq_backend.rs`.
9. `ipc:///tmp/spikenaut_readout.ipc` default endpoint constant in `src/zmq_backend.rs`.

Additional cleanup targets discovered in metadata/docs:

- `Cargo.toml` TODO note in repository field.
- README references to neural/hybrid-specific naming in public API lists.

## Public API target (planning)

Target API shape for a generic runtime/IPC core:

- Keep transport-neutral trait boundary, but rename domain-coded trait/method identifiers.
- Keep canonical protocol structs, but migrate to neutral message naming conventions.
- Keep optional feature-gated transport adapters, but use neutral transport naming.
- Keep crate-root exports, but avoid product-coded type names in exported surface.

Representative naming direction (non-binding planning examples):

- `NeuralBackend` -> `RuntimeBackend`
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

## Open questions

1. Should wire-format stability be preserved via serde aliases during renaming, or is a versioned schema bump acceptable?
2. Should runtime trait abstractions and wire models remain in one crate, or be split into `core-schema` + `runtime-adapters` crates?
3. Which binaries (if any) remain in this repository versus moving into app/supervisor repos?
4. What is the deprecation window for legacy names before full removal?

## Proposed migration sequence (planning only)

1. Approve boundary + naming matrix in issue #3.
2. Finalize neutral terminology map (old -> new) and compatibility policy, including **all `SPIKENAUT_*` env vars and default endpoint strings**.
3. Stage non-breaking docs and type alias/deprecation pass.
4. Execute major rename pass with release notes and downstream coordination.
5. Remove deprecated legacy names after agreed window.
