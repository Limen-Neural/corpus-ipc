# Plan: Clean legacy naming for GitHub #5 (NERO/Brain/Spine + trader terms)

**GitHub issue**: https://github.com/Limen-Neural/corpus-ipc/issues/5  
**Branch**: `chore/5-clean-legacy-naming`  
**Strategy**: Direct hard rename (no deprecation aliases, no `#[serde(rename)]`). Matches the hard-cut approach used for service/env vars in #12.  
**Scope**: Broad (boundary plan's four items + extras: bull/bear channels, brain_tick).  
**Target names (ultra-short neutral)**:
- `NeroManifoldSnapshot` → `Snapshot`
- `ZmqBrainBackend` → `ZmqBackend`; `BackendType::ZmqBrain` → `BackendType::Zmq`
- `SpineMessage` → `Message` (inner variants unchanged)
- `process_signals(...)` → `process(...)`
- `brain_tick` / `brain_tick()` → `tick` / `tick()`
- "bull channel" / "bear channel" (comments + test names) → "positive channel" / "negative channel"

**Wire impact**: Hard change to serde-visible names for `Snapshot` and `Message`. ZMQ binary packet format is unaffected (hand-parsed bytes).

## Goals
- Remove legacy NERO/trader/neural product-domain names from the public API surface and internal code.
- Align with the "generic runtime/IPC library" boundary defined in `docs/plans/corpus-ipc-boundary-plan.md`.
- Keep the crate functional and CI-clean after the renames.

## Non-goals (explicitly out of scope for this branch)
- Splitting the crate into core-schema + adapters.
- Adding serde compatibility shims or deprecation periods.
- Renaming variants inside `Message` (Spikes, Embeddings, etc.).
- Changes outside this repository.

## Ordered task list

1. Create branch `chore/5-clean-legacy-naming` from current `main`.

2. Perform the renames (one logical group at a time, then verify):
   - `src/models.rs`: `NeroManifoldSnapshot` → `Snapshot` (struct, impl, `from_scores`, all docs/comments that say "NERO").
   - `src/lib.rs`: update re-export and doc comment.
   - `src/trait_def.rs`: `BackendType::ZmqBrain` → `Zmq`; update doc comments.
   - `src/zmq_backend.rs`: `ZmqBrainBackend` → `ZmqBackend` (struct, impls, `new`, module docs, logs, tests); `brain_tick` field + `brain_tick()` + `pub brain_tick` → `tick` + `tick()`.
   - `src/trait_def.rs` + `src/bin/corpus_ipc_server.rs`: update `BackendType::Zmq` usage and `ZmqBackend` construction.
   - `src/trait_def.rs`: `process_signals` → `process` (trait method + docs).
   - `src/rust_backend.rs`: `process_signals` → `process` (impl + all call sites + test bodies); update "bull"/"bear" comments and test names (`positive_input_goes_to_positive_channel`, `negative_input_goes_to_negative_channel`); fix any inline comments that reference channels.
   - `src/error.rs`, `src/zmq_backend.rs`, `src/rust_backend.rs`: update any remaining `process_signals` mentions in docs.
   - Update `lib.rs` re-exports and the deprecated `NeuralBackend` alias doc if it mentions old names.
   - Update `README.md`: all mentions of old names, crate exports list, quickstart if needed.
   - Update `docs/plans/corpus-ipc-boundary-plan.md`: remove the four items from "Legacy wording inventory", update status for #5 items, note the additional cleanups performed.
   - Update `CHANGELOG.md` under `[Unreleased]` → `### Changed`: one bullet citing the renames and #5.

3. Run validation (must all pass before PR):
   - `cargo fmt -- --check`
   - `cargo clippy --all-targets --all-features -- -D warnings` (exact CI command)
   - `cargo test --all-features`
   - `cargo build --all-features --bins`
   - Confirm no remaining occurrences of the old strings in the repo (use grep for `Nero`, `BrainBackend`, `SpineMessage`, `process_signals`, `bull channel`, `bear channel`, `brain_tick` outside comments that are intentionally historical).

4. Commit with message that cites the issue and uses the required attribution style from prior work (e.g. "— Kilo Code agent..." if following previous convention).

5. Open PR from the branch targeting `main`. Title example:
   `chore: clean legacy naming (Nero/Brain/Spine/process_signals + bull/bear) (#5)`

6. Post-merge actions:
   - Add a comment on GitHub issue #5 linking the PR and stating completion.
   - Close #5 as completed.
   - (Optional but recommended) Update the boundary plan status line if it tracks multiple issues.

## Affected files (known from current tree)
- `src/models.rs`
- `src/lib.rs`
- `src/trait_def.rs`
- `src/zmq_backend.rs`
- `src/rust_backend.rs`
- `src/bin/corpus_ipc_server.rs`
- `src/error.rs` (docs only)
- `README.md`
- `CHANGELOG.md`
- `docs/plans/corpus-ipc-boundary-plan.md`

Tests that will change: `positive_input_goes_to_bull_channel`, `negative_input_goes_to_bear_channel`, and any assertions or comments mentioning channels or `brain_tick`.

## Risks / notes
- This is a breaking change for any downstream code using the old identifiers or relying on serde JSON keys for `Snapshot` / `Message`.
- No legacy aliases will be kept (consistent with prior hard renames).
- The ZMQ wire packet layout is unchanged; only Rust type names and JSON serde forms move.
- `HybridFlowBackend` trait methods are not renamed in this pass (they are already neutral).

## Validation checklist (for the implementer)
- [ ] `cargo fmt -- --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features` (all tests green, including renamed ones)
- [ ] `cargo build --all-features --bins`
- [ ] CHANGELOG entry added
- [ ] Boundary plan doc updated
- [ ] GitHub #5 comment posted + issue closed after merge
- [ ] No stray occurrences of old legacy names left in source/docs

## Open questions at plan time
None — all major decisions (scope, names, strategy, serde impact, extra terms, validation, branch style) were resolved during planning.

---
Plan created for issue #5. Ready for implementation on branch `chore/5-clean-legacy-naming`.
