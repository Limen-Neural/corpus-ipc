# AGENTS.md

See `README.md` for the crate overview, protocol models, and public API.

## Repository verification script (`scripts/verify.sh`)

`scripts/verify.sh` is the repo-owned check runner. It mirrors the gates in
`.github/workflows/ci.yml` and never weakens them; it is repo-only tooling and
stays out of the crates.io package (verify with `cargo package --list --locked`).
Every cargo command it runs uses `--locked` because `Cargo.lock` is tracked.

Two modes:

- `scripts/verify.sh` or `scripts/verify.sh --mode fast` (default) — fast dev
  checks, **no C++ toolchain required**: `cargo fmt --check` (skip with
  `--no-fmt`), `cargo check --no-default-features --locked`, `cargo check
  --features server --locked`, and the dependency-boundary assertion that the
  default `cargo tree --no-default-features --edges normal` graph excludes
  `axum`/`tokio`/`tower`/`zmq`. Cheap enough to run on every change.
- `scripts/verify.sh --mode full` — adds the complete release/CI qualification
  and **needs a working C++ compiler** (the `--all-features` path compiles
  vendored ZeroMQ from C++ via `zmq-sys`): `cargo clippy --all-targets
  --all-features --locked -- -D warnings`, `cargo test --all-features --locked`,
  `cargo test -p ap-check --locked`, `RUSTDOCFLAGS="-D warnings" cargo doc
  --no-deps --all-features --locked`, then `cargo package --list --locked`,
  `cargo package --locked`, and `cargo publish --dry-run --locked` run once at
  the end. It does not `cargo clean` between phases, so artifacts are reused.

Each phase prints a labeled banner (`==> [full 5/10] ...`); any failure exits
non-zero and names the failing phase. There are no error-swallowing constructs
and no silent skips. If the full-mode `--all-features` build fails with a C++
`cc-rs` error (e.g. `fatal error: 'string' file not found`) rather than a
corpus-ipc code failure, that is an environment gap, not a regression — re-run
with gcc/g++:

```bash
CC=gcc CXX=g++ scripts/verify.sh --mode full
```

Parallelism: the `--all-features`/`zmq` build is the parallelism-heavy phase
(see `docs/compile-cost.md`: ~86.6 s at `-j1` vs ~15.8 s at `-j8` on an 8-core
box). An **opt-in** cap (`--jobs N` or `CARGO_BUILD_JOBS=N`) is applied only to
the heavy full-mode phases; the fast path always runs at full `nproc` and is
never capped by default.

Observed verification budget (environment-specific; measured on this sandbox's
cargo/rustc 1.92.0, not the pinned-CI 1.98.1 toolchain — re-measure on the
target runner): fast checks are single-digit seconds cold and effectively free
warm; a cold `--all-features` build is ~16 s (up to ~37 s on the first-ever cold
C++ file cache). Full numbers, method, and caveats are in
[`docs/compile-cost.md`](docs/compile-cost.md) (repo-only; not packaged).

### Setup / maintenance (no provider-specific secrets)

These prime the cache and avoid redundant recompilation; none require secrets:

```bash
cargo fetch --locked        # prime the registry cache once (no compilation)
scripts/verify.sh           # fast per-feature checks (no C++ needed)
scripts/verify.sh --mode full   # full release/CI qualification (needs C++)
```

The retained external [Codex Cloud environment] is a **separate, owner-applied**
setup and is not created, changed, or tested by this repository change. An owner
with environment access can apply the commands above there; treat that as a
manual verification step. Do not claim the external environment was modified
without evidence.

[Codex Cloud environment]: https://chatgpt.com/codex/cloud/settings/environment/6a0beb131f8c8191979ea10df0b36d32

## Cursor Cloud specific instructions

Rust stable (edition-2024 capable) is preinstalled in the VM; the startup update script runs `cargo fetch`. Standard commands:

- Build / test / lint: `cargo build`, `cargo check --features zmq`, `cargo check --features server`, `cargo test --all-features`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo fmt --check`.
- `--all-features` enables both `zmq` and `server`. `zmq` builds ZeroMQ from **vendored C++ source** via `zmq-sys` (it does not link the system libzmq). This needs a working C++ compiler. The VM's `c++`/`cc` are set to **g++/gcc**, because the preinstalled clang cannot find libstdc++ headers. If a `fatal error: 'string' file not found` build error reappears (e.g. after a toolchain reset), restore it with:
  `CC=gcc CXX=g++ cargo build --all-features`
- Run the REST service (binary `corpus_ipc_server`, requires feature `server`):
  `CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features server --bin corpus_ipc_server`
  Add `zmq` as well (`--features server,zmq`) when selecting `CORPUS_IPC_BACKEND_TYPE=zmq`; otherwise the backend defaults to `Rust`. The process listens on `127.0.0.1:8080` when `CORPUS_IPC_BIND` is unset. Use `CORPUS_IPC_BIND` for another address; a remote bind (for example `0.0.0.0:8080`) needs an external access-control boundary because the router has no authentication or TLS. Exercise the service with `POST /initialize`, `POST /process {"inputs":[...]}`, `POST /save_state {"model_path":"..."}`, and `POST /reset`.
