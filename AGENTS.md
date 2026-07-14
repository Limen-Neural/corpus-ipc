# AGENTS.md

See `README.md` for the crate overview, protocol models, and public API.

## Cursor Cloud specific instructions

Rust stable (edition-2024 capable) is preinstalled in the VM; the startup update script runs `cargo fetch`. Standard commands:

- Build / test / lint: `cargo build`, `cargo test --all-features`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo fmt --check`.
- `--all-features` enables the `zmq` feature, which builds ZeroMQ from **vendored C++ source** via `zmq-sys` (it does not link the system libzmq). This needs a working C++ compiler. The VM's `c++`/`cc` are set to **g++/gcc**, because the preinstalled clang cannot find libstdc++ headers. If a `fatal error: 'string' file not found` build error reappears (e.g. after a toolchain reset), restore it with:
  `CC=gcc CXX=g++ cargo build --all-features`
- Run the REST service (binary `corpus_ipc_server`, auto-discovered from `src/bin/`):
  `CORPUS_IPC_BIND=127.0.0.1:8080 cargo run --release --features zmq --bin corpus_ipc_server`
  The `zmq` feature is required when selecting `CORPUS_IPC_BACKEND_TYPE=zmq`; otherwise the backend defaults to `Rust`. Use `CORPUS_IPC_BIND` to opt into a different address, including intentional remote access. Exercise the service with `POST /initialize`, `POST /process {"inputs":[...]}`, `POST /save_state {"model_path":"..."}`, and `POST /reset`.
