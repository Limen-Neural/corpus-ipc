#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# corpus-ipc repository verification runner (LIM-1387).
#
# Repository-owned check path that mirrors the gates already enforced in
# .github/workflows/ci.yml. It complements CI; it never replaces or weakens it.
# This script is repo-only tooling and intentionally stays OUT of the
# Cargo.toml `include` allow-list, so it is never shipped to crates.io (verify
# with `cargo package --list --locked`).
#
# Two clearly separated modes:
#
#   fast  (default)  Per-feature checks that need NO C++ toolchain. Cheap enough
#                    to run on every change: default + server `cargo check`, plus
#                    the dependency-boundary assertion that the default graph
#                    excludes axum/tokio/tower/zmq. Mirrors the `validate` job's
#                    fast per-feature checks.
#   full             Everything in `release-qualification`: adds a discrete
#                    `cargo check --features zmq` (matching the `validate` job's
#                    standalone zmq check), then all-targets / all-features
#                    clippy, all-features tests, the ap-check canonical-encoding
#                    test, rustdoc (-D warnings), and the package / publish
#                    --dry-run pair (run once, last). The zmq and all-features
#                    paths compile vendored ZeroMQ from C++, so this mode needs a
#                    working C++ compiler.
#
#                    Full mode does NOT re-run ci.yml `validate`'s discrete
#                    per-feature `cargo build --features server`, `cargo build
#                    --all-features`, and `cargo test --features server` steps:
#                    the all-features clippy/test phases are a strict superset
#                    that compiles and tests every feature, so re-running the
#                    per-feature builds would only repeat heavy work the issue
#                    explicitly discourages without catching anything new.
#
# Every cargo invocation uses --locked because Cargo.lock is tracked.
#
# ---------------------------------------------------------------------------
# Parallelism note (evidence: docs/compile-cost.md, LIM-1387 FEAT-001)
# ---------------------------------------------------------------------------
# The all-features / zmq build is the parallelism-heavy phase: a cold
# all-features build measured ~86.6 s at -j1 versus ~15.8 s at -j8 (nproc) on an
# 8-core / 30 GiB sandbox (~5.5x). The fast per-feature checks are single-digit
# seconds cold and effectively free warm. Therefore any jobs cap here is:
#   * OPT-IN only (via --jobs N or CARGO_BUILD_JOBS); default is unbounded/nproc.
#   * applied ONLY to the heavy full-mode all-features phases (clippy/test/doc/
#     package/publish), never to the fast path.
# Capping the fast path would add no memory safety and only slow it down, so the
# fast path always runs at full parallelism.
# ---------------------------------------------------------------------------
#
# C++ / zmq environment note (see AGENTS.md / CLAUDE.md):
#   The `zmq` / `--all-features` build compiles vendored ZeroMQ from C++ via
#   zmq-sys; it does not link system libzmq. It needs a working C++ compiler.
#   If clang cannot find libstdc++ (`fatal error: 'string' file not found`),
#   re-run with `CC=gcc CXX=g++`. A full-mode failure in that phase surfaces the
#   real cargo/cc-rs error and points at this workaround; it is never auto-skipped.
#
# MSRV note:
#   The crate pins rust-version = 1.98.1 and CI installs exactly that toolchain.
#   This script targets a correct (>= MSRV) toolchain and never bakes in
#   `--ignore-rust-version`. On a runner whose rustc is older than the MSRV,
#   cargo will refuse the build/check/clippy/test phases by design.

# -E (errtrace) makes the ERR trap fire inside functions too, so a cargo failure
# in run_fast / run_full reports the failing phase instead of exiting silently.
set -Eeuo pipefail

# --- configuration ---------------------------------------------------------

MODE="fast"
# Empty means "let cargo use its default parallelism" (nproc). A positive value
# is passed as `--jobs N` ONLY to the heavy full-mode phases. CARGO_BUILD_JOBS,
# if exported, is honored by cargo natively; --jobs takes precedence when set.
JOBS="${CARGO_BUILD_JOBS:-}"
RUN_FMT=1

usage() {
    cat <<'EOF'
Usage: scripts/verify.sh [--mode fast|full] [--jobs N] [--no-fmt] [-h|--help]

Modes:
  fast   (default) Per-feature checks, no C++ toolchain required:
           cargo fmt --check (skip with --no-fmt)
           cargo check --no-default-features --locked
           cargo check --features server --locked
           dependency-boundary assertion (default tree excludes axum/tokio/tower/zmq)
  full   Adds the full release/CI qualification (needs a C++ compiler for
         zmq / --all-features):
           cargo check --features zmq --locked
           clippy --all-targets --all-features -D warnings,
           test --all-features, test -p ap-check, rustdoc -D warnings,
           package --list, package, publish --dry-run.

Options:
  --jobs N   Opt-in Cargo parallelism cap applied ONLY to the heavy full-mode
             all-features phases. Default: unbounded (nproc). The fast path is
             never capped. Equivalent to exporting CARGO_BUILD_JOBS=N.
  --no-fmt   Skip the `cargo fmt --check` phase (fmt is a formatting gate, not a
             compile gate; it stays on by default to mirror CI).
  -h, --help Show this help.

Every cargo command uses --locked. See docs/compile-cost.md for the observed
verification budget and AGENTS.md / CLAUDE.md for the C++ / zmq build note.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || { echo "error: --mode requires an argument (fast|full)" >&2; exit 2; }
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --jobs)
            [[ $# -ge 2 ]] || { echo "error: --jobs requires a positive integer" >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        --no-fmt)
            RUN_FMT=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    fast|full) ;;
    *)
        echo "error: --mode must be 'fast' or 'full' (got '$MODE')" >&2
        exit 2
        ;;
esac

if [[ -n "$JOBS" ]]; then
    if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: --jobs / CARGO_BUILD_JOBS must be a positive integer (got '$JOBS')" >&2
        exit 2
    fi
fi

# Run from the repository root so cargo and relative paths resolve regardless of
# the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- phase machinery -------------------------------------------------------

PHASE_TOTAL=0
PHASE_INDEX=0
CURRENT_PHASE=""

banner() {
    # banner "<label>"  -> prints "==> [<mode> N/TOTAL] <label>"
    PHASE_INDEX=$((PHASE_INDEX + 1))
    CURRENT_PHASE="$1"
    echo ""
    echo "==> [${MODE} ${PHASE_INDEX}/${PHASE_TOTAL}] ${CURRENT_PHASE}"
}

# Report the failing phase on ANY non-zero exit (set -e / pipefail included).
# Errors are never swallowed and cargo failures are never disabled: every phase
# fails loudly with the real cargo error already printed above this message.
on_error() {
    local code=$?
    echo ""
    echo "xx FAILED in ${MODE} mode during phase ${PHASE_INDEX}/${PHASE_TOTAL}: ${CURRENT_PHASE}" >&2
    echo "xx exit code: ${code}" >&2
    if [[ "$MODE" == "full" && "$CURRENT_PHASE" == *all-features* ]]; then
        echo "xx note: --all-features compiles vendored ZeroMQ from C++ (zmq-sys)." >&2
        echo "xx       If this is a C++/cc-rs error (e.g. \"'string' file not found\")" >&2
        echo "xx       rather than a corpus-ipc code failure, retry with a working" >&2
        echo "xx       C++ compiler: CC=gcc CXX=g++ scripts/verify.sh --mode full" >&2
        echo "xx       (see AGENTS.md / CLAUDE.md). The phase is never auto-skipped." >&2
    fi
    exit "$code"
}
trap on_error ERR

# --- dependency-boundary assertion (shared with ci.yml logic) --------------
# Mirrors the "Assert default tree excludes HTTP and ZMQ stacks" step in
# .github/workflows/ci.yml: the default (no-default-features) normal-edge graph
# must never contain axum, tokio, tower, or zmq.
assert_default_tree_boundary() {
    local tree
    tree=$(cargo tree --no-default-features --edges normal --locked)
    echo "$tree"
    if echo "$tree" | grep -E '(^|[[:space:]])(axum|tokio|tower|zmq) v'; then
        echo "default feature graph must not include axum, tokio, tower, or zmq" >&2
        return 1
    fi
    echo "boundary holds: default graph excludes axum/tokio/tower/zmq"
}

# --- fast mode -------------------------------------------------------------

run_fast() {
    # Phase count depends on whether fmt runs.
    if [[ "$RUN_FMT" -eq 1 ]]; then
        PHASE_TOTAL=4
        banner "cargo fmt --check"
        cargo fmt --check
    else
        PHASE_TOTAL=3
    fi

    banner "cargo check --no-default-features --locked"
    cargo check --no-default-features --locked

    banner "cargo check --features server --locked"
    cargo check --features server --locked

    banner "dependency-boundary assertion (cargo tree --no-default-features --edges normal --locked)"
    assert_default_tree_boundary
}

# --- full mode -------------------------------------------------------------

# Heavy all-features phases honor the opt-in jobs cap. Cheap fast-path phases do
# not. JOBS_ARGS is empty (full nproc) unless --jobs / CARGO_BUILD_JOBS is set.
JOBS_ARGS=()
if [[ -n "$JOBS" ]]; then
    JOBS_ARGS=(--jobs "$JOBS")
fi

run_full() {
    # Order: fast per-feature checks first (cheap, no C++), then the heavy
    # all-features qualification, then package/publish --dry-run ONCE at the end.
    # No `cargo clean` anywhere: artifacts are reused across phases.
    if [[ "$RUN_FMT" -eq 1 ]]; then
        PHASE_TOTAL=12
        banner "cargo fmt --check"
        cargo fmt --check
    else
        PHASE_TOTAL=11
    fi

    banner "cargo check --no-default-features --locked"
    cargo check --no-default-features --locked

    banner "cargo check --features server --locked"
    cargo check --features server --locked

    # Discrete zmq check, mirroring ci.yml `validate`'s standalone "Check zmq
    # feature" step. This compiles vendored ZeroMQ from C++ (zmq-sys), so it
    # lives in full mode, never in the C++-free fast mode.
    banner "cargo check --features zmq --locked"
    cargo check --features zmq --locked

    banner "dependency-boundary assertion (cargo tree --no-default-features --edges normal --locked)"
    assert_default_tree_boundary

    # Heavy phases below compile vendored ZeroMQ (C++). Opt-in jobs cap applies.
    banner "cargo clippy --all-targets --all-features --locked -- -D warnings"
    cargo clippy "${JOBS_ARGS[@]}" --all-targets --all-features --locked -- -D warnings

    banner "cargo test --all-features --locked"
    cargo test "${JOBS_ARGS[@]}" --all-features --locked

    banner "cargo test -p ap-check --locked"
    cargo test "${JOBS_ARGS[@]}" -p ap-check --locked

    banner "RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features --locked"
    RUSTDOCFLAGS="-D warnings" cargo doc "${JOBS_ARGS[@]}" --no-deps --all-features --locked

    # package/publish rebuild in a temp dir; run them once, last, after the other
    # full checks so their cost is paid a single time per full run.
    banner "cargo package --list --locked"
    cargo package --list --locked

    banner "cargo package --locked"
    cargo package "${JOBS_ARGS[@]}" --locked

    banner "cargo publish --dry-run --locked"
    cargo publish "${JOBS_ARGS[@]}" --dry-run --locked
}

# --- dispatch --------------------------------------------------------------

echo "corpus-ipc verify.sh: mode=${MODE}, fmt=$([[ $RUN_FMT -eq 1 ]] && echo on || echo off), jobs=${JOBS:-nproc}"

case "$MODE" in
    fast) run_fast ;;
    full) run_full ;;
esac

echo ""
echo "== ${MODE} mode: all ${PHASE_TOTAL} phase(s) passed."
