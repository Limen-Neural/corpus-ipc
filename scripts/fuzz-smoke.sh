#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Bounded cargo-fuzz smoke campaign for the three corpus-ipc targets (LIM-1277).
# Repo-only tooling: needs nightly Rust, cargo-fuzz, and a C++ compiler (the
# targets enable the `zmq` feature). Not part of default PR checks on every
# platform; CI runs it in the Linux-only `fuzz-smoke` workflow.
#
# Seeds come from the committed vectors in place (test-vectors/wire-v1 for the
# JSON targets; test-vectors/zmq hex fixtures decoded into a scratch dir for
# the readout target) plus fuzz/regressions/<target>. Nothing is copied into
# the repository: new inputs land in a scratch corpus that is discarded.
#
# Usage: scripts/fuzz-smoke.sh [--seconds N] [--toolchain NAME] [target...]
#   --seconds N        wall-clock budget per target (default 30)
#   --toolchain NAME   rustup toolchain for cargo-fuzz (default "nightly")
#   target...          subset of targets (default: all three)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seconds=30
toolchain=nightly
targets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds) seconds="$2"; shift 2 ;;
    --toolchain) toolchain="$2"; shift 2 ;;
    -h|--help) sed -n '3,19p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) targets+=("$1"); shift ;;
  esac
done
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(fuzz_ipc_message_json fuzz_zmq_readout_frame fuzz_zmq_message_frame)
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# Decode the hex readout fixtures to raw bytes (the fixtures stay authoritative).
mkdir -p "$scratch/zmq-seeds"
for hex in "$root"/test-vectors/zmq/*.hex; do
  python3 -c 'import sys; open(sys.argv[2], "wb").write(bytes.fromhex(open(sys.argv[1]).read().strip()))' \
    "$hex" "$scratch/zmq-seeds/$(basename "$hex" .hex).bin"
done

cd "$root/fuzz"
for target in "${targets[@]}"; do
  case "$target" in
    fuzz_ipc_message_json|fuzz_zmq_message_frame)
      seeds=("$root/test-vectors/wire-v1"); max_len=262144 ;;
    fuzz_zmq_readout_frame)
      seeds=("$scratch/zmq-seeds"); max_len=8192 ;;
    *) echo "unknown target: $target" >&2; exit 2 ;;
  esac
  corpus="$scratch/corpus-$target"
  mkdir -p "$corpus"
  echo "==> $target (${seconds}s, max_len=$max_len)"
  # First directory is the writable corpus; the rest are read-only seeds.
  # -max_len above 65_536 lets JSON inputs cross ProtocolLimits::DEFAULT.
  cargo "+$toolchain" fuzz run --release "$target" \
    "$corpus" "${seeds[@]}" "$root/fuzz/regressions/$target" -- \
    -max_total_time="$seconds" -max_len="$max_len" -timeout=10 -rss_limit_mb=2048 \
    -print_final_stats=1
done
echo "fuzz smoke passed: ${targets[*]}"
