<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc fuzz targets (LIM-1277)

Repository-only [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) /
libFuzzer harness. `fuzz/` is its own Cargo workspace, is not a member of the
root workspace, and sits outside the crate's `include` allow-list, so the
published `corpus-ipc` package gains no libFuzzer dependency, binary, or file
(check with `cargo package --list --locked`).

## Targets

Every target calls a production entry point. No parser is copied.

| Target | Entry point | Invariants |
| --- | --- | --- |
| `fuzz_ipc_message_json` | `decode_ipc_message_json` (compatibility decoder) | No panic. An accepted message passes `Validate` at `ProtocolLimits::DEFAULT`, re-encodes with `encode_canonical_ipc_message`, decodes back to the same message, and re-encodes to identical bytes. |
| `fuzz_zmq_readout_frame` | `zmq_readout::parse_readout_packet` and the `ZmqIpcBackend` ingest path built on it | No panic. Accepted iff `len ≥ 8`, `(len − 8) % 4 == 0`, and the float count is within the cap (checked at the default 1024 and at 16). Decoded tick and floats match the little-endian bytes bit for bit. The backend cache changes only when the parser accepts. |
| `fuzz_zmq_message_frame` | `ZmqMessageSubscriber::decode_frame_for_tests`, the decoder `try_recv` runs on one frame | No panic. Empty frame → `UnexpectedFraming(Empty)`. Other rejections are only `Envelope` or `Validation` errors. Accepted messages agree with `decode_ipc_message_json`, pass `Validate`, round-trip canonically, and fit `STRUCTURED_MAX_FRAME_BYTES`. |

The invariants live in [`src/lib.rs`](src/lib.rs). The root crate compiles the
same file into [`tests/fuzz_regressions.rs`](../tests/fuzz_regressions.rs), so
regression seeds, the committed vectors, and synthetic empty / truncated /
random / oversize inputs replay on stable `cargo test --all-features` without
libFuzzer.

## Running

Needs nightly Rust, `cargo install cargo-fuzz --locked`, and a C++ compiler
(the targets enable feature `zmq`, which builds vendored libzmq).

```bash
scripts/fuzz-smoke.sh                       # all targets, 30 s each
scripts/fuzz-smoke.sh --seconds 300 fuzz_zmq_message_frame
scripts/fuzz-smoke.sh --toolchain nightly-2026-10-19   # the CI pin
```

The script seeds each campaign from the committed vectors **in place**:
`test-vectors/wire-v1/` for both JSON targets, and `test-vectors/zmq/*.hex`
decoded into a scratch directory for the readout target. Those fixtures stay
the authoritative compatibility corpus; nothing here duplicates them. New
inputs go to a scratch corpus that is deleted on exit. JSON campaigns use
`-max_len=262144` so inputs can cross `ProtocolLimits::DEFAULT`; every run is
capped by `-max_total_time`, `-timeout=10`, and `-rss_limit_mb=2048`.

CI runs the script for 60 s per target in the Linux-only
[`Fuzz smoke`](../.github/workflows/fuzz.yml) workflow. The cross-platform `CI`
workflow never needs libFuzzer.

## Regression seeds

`regressions/<target>/` holds minimized inputs, one file per fixed crash or
invariant violation. Both the smoke script (as a seed directory) and
`tests/fuzz_regressions.rs` replay them. When a campaign finds a failure:

1. Minimize it: `cargo +nightly fuzz tmin <target> fuzz/artifacts/<target>/crash-…`.
2. Fix the production code.
3. Commit the minimized input as `regressions/<target>/<short-name>` and add a
   row to the table below naming the originating failure and the fix.

| Target | Seed | Originating failure | Fix |
| --- | --- | --- | --- |
| — | — | No crash or invariant violation found yet (see below). | — |

Initial campaigns on 2026-10-19 (nightly-2026-10-19, release build):
3 × 60 s and 3 × 600 s per target, seeded from the committed vectors, found no
panic, timeout, out-of-memory, or invariant violation, so no regression seed
exists yet. The directories are kept (with `.gitkeep`) so a first finding has a
place to land.
