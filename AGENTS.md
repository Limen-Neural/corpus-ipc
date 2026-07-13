# AGENTS.md

See `README.md` for the crate overview, protocol models, and public API.

## Cursor Cloud specific instructions

Rust stable (edition-2024 capable) is preinstalled in the VM; the startup update script runs `cargo fetch`. Standard commands:

- Build / test / lint: `cargo build`, `cargo test --all-features` (8 tests), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo fmt --check`.
- `--all-features` enables the `zmq` feature, which builds ZeroMQ from **vendored C++ source** via `zmq-sys` (it does not link the system libzmq). This needs a working C++ compiler. The VM's `c++`/`cc` are set to **g++/gcc**, because the preinstalled clang cannot find libstdc++ headers. If a `fatal error: 'string' file not found` build error reappears (e.g. after a toolchain reset), restore it with:
  `sudo update-alternatives --auto c++ && sudo update-alternatives --auto cc`
- Run the REST service (binary `corpus_ipc_server`, auto-discovered from `src/bin/`):
  `cargo run --release --bin corpus_ipc_server`
  It binds `0.0.0.0:8080` (override with `CORPUS_IPC_BIND`); backend selected by `CORPUS_IPC_BACKEND_TYPE` (default `Rust`). Exercise it with `POST /initialize`, `POST /process {"inputs":[...]}`, `POST /reset`.
