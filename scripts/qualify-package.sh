#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Qualify the packaged crate (not the checkout): package, extract outside the
# repo, build/test the extraction per feature set, build three independent
# consumers against it, then rustdoc and `cargo publish --dry-run`.
# Never uploads. Repo-only tooling; excluded from the crates.io package.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/corpus-ipc-qualify.XXXXXX")"
trap 'rm -rf "$work"' EXIT
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d" " -f1; }
banner() { printf '\n==== %s ====\n' "$*"; }

cd "$root"
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing: working tree is dirty (qualify an exact commit)" >&2
  exit 1
fi
sha="$(git rev-parse HEAD)"
name="$(cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json,sys;p=[p for p in json.load(sys.stdin)["packages"] if p["name"]=="corpus-ipc"][0];print(p["name"]+"-"+p["version"])')"

banner "release candidate $sha ($name)"
banner "package inventory"
cargo package --locked --list | tee "$work/inventory.txt"
if grep -E '^(\.claude|\.github|\.vscode|\.idea|AGENTS\.md|CLAUDE\.md|target/|fuzz/|ci/|scripts/|\.cursor)' "$work/inventory.txt"; then
  echo "repo-only state leaked into the package" >&2; exit 1
fi
for f in LICENSE-MIT LICENSE-APACHE-2.0 README.md CHANGELOG.md docs/wire-encoding.md docs/rest-api.md; do
  grep -qx "$f" "$work/inventory.txt" || { echo "missing from package: $f" >&2; exit 1; }
done

banner "cargo package --locked (verifies by building)"
cargo package --locked
crate="$root/target/package/$name.crate"
echo "sha256: $(sha256 "$crate")"
echo "size:   $(wc -c < "$crate" | tr -d " ") bytes"

banner "extract outside the checkout"
mkdir "$work/ex"
tar -xzf "$crate" -C "$work/ex"
pkg="$work/ex/$name"
export CARGO_TARGET_DIR="$work/target-ex"
cd "$pkg"
# The packaged lockfile pins versions; drop the workspace so it stands alone.
for f in "" "--no-default-features" "--features zmq" "--features server" "--all-features"; do
  banner "extracted: build+test ${f:-default}"
  # shellcheck disable=SC2086
  cargo test --locked $f
done

banner "extracted: rustdoc -D warnings"
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

mkcons() { # dir features
  mkdir -p "$work/$1/src"
  cat > "$work/$1/Cargo.toml" <<TOML
[package]
name = "$1"
version = "0.0.0"
edition = "2024"
publish = false
[dependencies]
corpus-ipc = { path = "$pkg"${2:+, features = [$2]} }
$3
[workspace]
TOML
}

banner "consumer: schema-only/default"
mkcons consumer-default "" 'serde_json = "1"'
cat > "$work/consumer-default/src/main.rs" <<'RS'
use corpus_ipc::{
    BackendFactory, BackendType, IpcBackend, IpcMessage, ProtocolLimits, RustBackend,
    StimulusBatch, Validate, decode_ipc_message_json, encode_canonical_ipc_message,
};
fn main() {
    let s: StimulusBatch = serde_json::from_str(
        r#"{"session_id":null,"batch_id":1,"timestamp":2,"values":[1.0,-1.0],"valid_mask":null,"metadata":null}"#,
    )
    .unwrap();
    s.validate_with(ProtocolLimits::DEFAULT).unwrap();
    let msg = IpcMessage::Stimuli(s);
    let bytes = encode_canonical_ipc_message(&msg).unwrap();
    assert!(decode_ipc_message_json(&bytes).is_ok());
    let mut b: Box<dyn IpcBackend> = BackendFactory::create(BackendType::Rust);
    b.initialize(None).unwrap();
    let _ = RustBackend::new();
    println!("default consumer ok");
}
RS
(cd "$work/consumer-default" && cargo run --quiet)

banner "consumer: zmq (structured + binary)"
mkcons consumer-zmq '"zmq"' ""
cat > "$work/consumer-zmq/src/main.rs" <<'RS'
use corpus_ipc::{
    BackendFactory, BackendType, FramingFault, HybridFlowBackend, STRUCTURED_MAX_FRAME_BYTES,
    ZmqHybridFlowBackend, ZmqIpcBackend, ZmqMessageError, ZmqMessagePublisher,
    ZmqMessageSubscriber, ZmqTransportConfig,
};
use corpus_ipc::zmq_readout;
fn main() {
    let cfg = ZmqTransportConfig::new("inproc://qualify");
    let _publisher = ZmqMessagePublisher::bind(cfg.clone()).unwrap();
    let sub = ZmqMessageSubscriber::connect(cfg).unwrap();
    let _: Result<_, ZmqMessageError> = sub.try_recv();
    let _ = STRUCTURED_MAX_FRAME_BYTES;
    let _: Option<FramingFault> = None;
    let _: Option<ZmqHybridFlowBackend> = None;
    fn _t<T: HybridFlowBackend>() {}
    _t::<ZmqHybridFlowBackend>();
    let _ = ZmqIpcBackend::new();
    let _ = BackendFactory::create(BackendType::ZmqIpc);
    let _ = zmq_readout::DEFAULT_MAX_READOUT_FLOATS;
    println!("zmq consumer ok");
}
RS
(cd "$work/consumer-zmq" && cargo run --quiet)

banner "consumer: server (library router + feature boundary)"
mkcons consumer-server '"server"' 'tokio = { version = "1", features = ["rt"] }'
cat > "$work/consumer-server/src/main.rs" <<'RS'
use corpus_ipc::server::{AppState, router};
use corpus_ipc::{BackendFactory, BackendType};
fn main() {
    let _app = router(BackendFactory::create(BackendType::Rust));
    let _ = AppState::new(BackendFactory::create(BackendType::Rust));
    println!("server consumer ok");
}
RS
(cd "$work/consumer-server" && cargo run --quiet)

banner "boundary: default consumer must NOT see server/zmq"
cat > "$work/consumer-default/src/main.rs" <<'RS'
use corpus_ipc::server::router;
fn main() { let _ = router; }
RS
if (cd "$work/consumer-default" && cargo check --quiet 2>/dev/null); then
  echo "server module leaked into default build" >&2; exit 1
fi
echo "boundary ok"

banner "cargo publish --dry-run --locked (checkout, exact commit)"
cd "$root"
unset CARGO_TARGET_DIR
cargo publish --dry-run --locked

banner "qualified $sha $name sha256=$(sha256 "$crate")"
