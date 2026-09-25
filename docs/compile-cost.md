<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc compile-cost measurements

Baseline evidence for the feature-gated compile paths, gathered for LIM-1387 so
the repository verification workflow can be planned against real numbers instead
of guesses. These are **observed, environment-specific** figures. They are **not**
a universal timeout or memory promise, and they are **not** the pinned-CI
toolchain result. Re-measure on the target runner before treating any number as a
budget there.

This document is repository-owned and intentionally stays **out of** the
`Cargo.toml` `include` allow-list, so it is never shipped to crates.io (only
`docs/wire-encoding.md` ships from `docs/`). Confirm with
`cargo package --list --locked`.

## Runner / toolchain (this measurement environment)

| Fact | Value |
| --- | --- |
| cargo | `1.92.0 (344c4567c 2025-10-21)` |
| rustc | `1.92.0 (ded5c06cf 2025-12-08)` |
| CPU cores (`nproc`) | 8 |
| RAM (`free -h`, total) | 30 GiB (≈28 GiB available at measure time) |
| OS | Amazon Linux 2023, kernel `6.1.x` x86_64 |
| C++ toolchain | gcc / g++ 11.5.0 (`cc`/`c++` resolve; clang also present) |
| Network | Available (crates.io downloads succeed) |

### Toolchain caveat (read this before using any number)

- The crate pins `rust-version = "1.98.1"` (MSRV) in `Cargo.toml`. This sandbox
  ships **cargo/rustc 1.92.0**, which is numerically **older** than that MSRV.
  Cargo therefore **refuses** a plain build:

  ```text
  error: rustc 1.92.0 is not supported by the following package:
    corpus-ipc@0.1.0 requires rustc 1.98.1
  ```

- To obtain build/check timing evidence at all, the measurements below were run
  with `--ignore-rust-version` appended to the `cargo build`/`cargo check`
  commands. That flag **only bypasses the manifest gate for the current
  invocation**; it does **not** modify the crate's MSRV and nothing in the repo
  was changed. The source compiles cleanly on 1.92.0 once the gate is bypassed.
- CI pins the toolchain to **1.98.1** via `dtolnay/rust-toolchain`. Compile
  numbers here are from a **different, older** toolchain and will differ from the
  pinned-CI runner. Treat every timing below as "observed on this sandbox with
  1.92.0", not as the CI budget.
- `cargo tree` and `cargo package --list` do **not** enforce `rust-version`, so
  the crate-count and package-list evidence needs no bypass and is unaffected by
  the toolchain mismatch.

## Dependency-graph contributors (per feature path)

Unique crate counts from `cargo tree` (normal edges only), deduplicated by
crate name. The `sed 's/ v.*//'` step strips the version suffix (and `cargo
tree`'s `(*)` "subtree already shown" markers) so a crate that appears more than
once collapses to a single name before `sort -u` counts it:

```bash
cargo tree --no-default-features --edges normal --locked --prefix none | sed 's/ v.*//' | sort -u | grep -v '^$' | wc -l
cargo tree --features server      --edges normal --locked --prefix none | sed 's/ v.*//' | sort -u | grep -v '^$' | wc -l
cargo tree --features zmq         --edges normal --locked --prefix none | sed 's/ v.*//' | sort -u | grep -v '^$' | wc -l
cargo tree --all-features         --edges normal --locked --prefix none | sed 's/ v.*//' | sort -u | grep -v '^$' | wc -l
```

| Path | Command (features) | Unique normal-edge crates |
| --- | --- | --- |
| default (no-default) | `--no-default-features` | 14 |
| server | `--features server` | 47 |
| zmq | `--features zmq` | 18 |
| all-features | `--all-features` | 50 |

The 14 crates in the default (no-default-features) normal-edge graph are:
`corpus-ipc`, `itoa`, `memchr`, `proc-macro2`, `quote`, `serde`, `serde_core`,
`serde_derive`, `serde_json`, `syn`, `thiserror`, `thiserror-impl`,
`unicode-ident`, `zmij`.

Observations:

- **default** is deliberately lean: `serde`, `serde_json` (+ `raw_value`
  support), `thiserror`, and their proc-macro/build deps. No `axum`, `tokio`,
  `tower`, or `zmq`.
- **server** adds the Axum/Tokio/Tower HTTP stack (+33 crates over default) and
  dominates the *crate-count* dimension.
- **zmq** adds only the vendored ZeroMQ FFI chain (`zmq`, `zmq-sys`, `libc`,
  `bitflags`) for +4 crates, but its cost is **native C++ compilation**, not
  crate count (see build timings).
- **all-features** = server ∪ zmq = 50 crates.

### Feature-boundary assertion

The default graph must never contain the server/zmq stacks. Verified:

```bash
cargo tree --no-default-features --edges normal --locked | grep -iE 'axum|tokio|tower|zmq'
# -> no matches (boundary holds)
```

## Build / check wall-clock

Timing method: `/usr/bin/time` is **not** available in this sandbox, so peak-RSS
via `/usr/bin/time -v` could not be captured (see "Memory evidence" below).
Wall-clock was taken with shell `date +%s%N` around each invocation. "Cold" =
immediately after `cargo clean` (target wiped) with the **registry cache already
primed** by `cargo fetch --locked` (crates downloaded, not recompiled from a
fresh download). "Warm" = an immediate no-op re-run with an intact `target/`.

All commands below were run with `--locked --ignore-rust-version` (see toolchain
caveat).

### `cargo check` (fast per-feature path, no C++ needed for default/server)

```bash
cargo clean
cargo check --no-default-features --locked   # cold
cargo check --no-default-features --locked   # warm (no-op)
cargo clean
cargo check --features server --locked       # cold
cargo check --features server --locked       # warm (no-op)
```

| Path | Cold check | Warm (no-op) check |
| --- | --- | --- |
| default | ~3.2 s (3209 ms) | ~0.03 s (33 ms) |
| server | ~6.6 s (6586 ms) | ~0.04 s (43 ms) |

### `cargo build`

```bash
cargo clean
cargo build --no-default-features --locked   # cold / warm
cargo build --features server --locked       # cold / warm
cargo build --features zmq --locked          # cold / warm
cargo build --all-features --locked          # cold / warm
```

| Path | Cold build | Warm (no-op) build |
| --- | --- | --- |
| default | ~3.8 s (3836 ms) | ~0.03 s (33 ms) |
| server | ~9.0 s (8972 ms) | ~0.04 s (44 ms) |
| zmq | ~12.9 s warm-FS / ~37.3 s first-ever (see note) | ~0.05 s (54 ms) |
| all-features | ~15.9 s (15828–15970 ms) | ~0.05 s (54 ms) |

**zmq first-build note.** The very first `cargo build --features zmq` in a fresh
environment measured **37.3 s** because `zmq-sys` compiles vendored ZeroMQ from
C++ (`zeromq-src`) with a completely cold compiler/header OS file cache. A
repeated cold-target build (target wiped, but OS file cache warm for the C++
sources/headers) measured **12.9 s**. Both are legitimate; the 37 s figure is the
true first-touch cost and the ~13 s figure is the steady-state
clean-target-rebuild cost on the same box. `all-features` measured ~16 s cold
because by then the C++ file cache was already warm.

Both zmq and all-features built successfully with the **plain** system compiler
(gcc/g++ 11.5.0); the `CC=gcc CXX=g++` workaround documented in AGENTS.md /
CLAUDE.md was **not** required here. It remains the fallback if a runner's clang
cannot find `libstdc++`.

### Parallelism pressure (proxy for the memory-heavy phase)

`cargo build --timings` was captured for the all-features path; on stable 1.92.0
only the **HTML** report is produced (`--timings=json` is nightly-only), and it
lands under `target/cargo-timings/` which is gitignored and **not committed**.

A concrete, reproducible parallelism proxy is the serialized-vs-parallel delta on
the all-features (heaviest) path:

```bash
cargo clean && cargo build --all-features --locked -j1   # serialized
cargo clean && cargo build --all-features --locked       # default (8 jobs)
```

| all-features cold build | Wall-clock |
| --- | --- |
| `-j1` (serialized) | ~86.6 s (86613 ms) |
| default (`-j8`, `nproc`) | ~15.8 s (15828 ms) |

That is a ~5.5x speedup from parallelism, confirming **all-features / zmq is the
parallelism-heavy phase** while default/server are cheap. This is the evidence a
verification script should use if it ever considers an opt-in jobs cap: bound
only the all-features/zmq phase and leave the fast paths at full `nproc`, because
capping jobs there would sharply increase wall-clock without a measured memory
reason to do so.

## Memory evidence: what could and could not be measured

- **Peak RSS could NOT be measured directly.** `/usr/bin/time` is absent, so
  `/usr/bin/time -v` (the usual "Maximum resident set size" source) was
  unavailable. No trustworthy peak-memory number was obtained, and none is
  fabricated here.
- **What is available instead:** the `cargo build --timings` HTML report (peak
  concurrency / per-unit timeline) and the `-j1` vs `-j8` delta above, which
  bound *parallelism* pressure but not bytes. The box has 30 GiB RAM and no build
  path OOM'd or swapped during any measurement.
- **Do not** derive a memory budget or a hard timeout from this document. If a
  target environment needs an RSS ceiling, measure it there with a tool that can
  report peak RSS (e.g. `/usr/bin/time -v`, `cgroup` memory peak, or the runner's
  own metrics).

## Observed verification budget (this runner only)

Rough wall-clock envelopes to size a check workflow, from the numbers above.
Cold assumes a wiped `target/` with the registry already fetched; warm assumes an
intact `target/`.

| Workflow slice | Cold | Warm |
| --- | --- | --- |
| Fast dev checks: default + server `cargo check` + tree boundary | ~10 s | < 1 s |
| Full compile of all-features (`cargo build --all-features`) | ~16 s | ~0.05 s |
| First-ever zmq/all-features touch (cold C++ file cache) | up to ~37 s | n/a |

Interpretation: the per-feature fast path is single-digit seconds cold and
effectively free warm, so a fast/dev mode can run it on every change. The
all-features/zmq path costs the most (native C++), so a script should avoid
recompiling it or re-running `cargo package` / `cargo publish --dry-run`
redundantly, and reserve those for an explicit full/release mode. Numbers are
**observed on cargo/rustc 1.92.0 on this sandbox**; the pinned-CI 1.98.1 runner
will differ and must be measured separately.

## Reproduce

```bash
cargo fetch --locked                       # prime registry cache once
# crate counts
for f in "--no-default-features" "--features server" "--features zmq" "--all-features"; do
  cargo tree $f --edges normal --locked --prefix none | sed 's/ v.*//' | sort -u | grep -v '^$' | wc -l
done
# timing (append --ignore-rust-version only if the runner's rustc < MSRV 1.98.1)
cargo clean && cargo build --all-features --locked --timings   # HTML under target/cargo-timings (gitignored)
```
