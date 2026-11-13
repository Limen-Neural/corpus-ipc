# REVIEW.md

Review guidance for contributors and reviewers.

## MSRV pin rule

`rust-version` and the dev/CI toolchain pin are **allowed to differ**
(`rust-version <= toolchain channel`). They answer different questions:

- **`package.rust-version` in `Cargo.toml`** is the *true measured minimum
  supported Rust version* — the oldest rustc that actually builds the crate.
  It is what crates.io advertises and what downstream consumers can rely on.
  Verify changes to it with `cargo +<ver> check` on the candidate toolchain;
  never raise it to match the dev toolchain without a measured reason.
- **`rust-toolchain.toml` (`channel`) and every non-MSRV CI `toolchain:` pin**
  are the *development toolchain*, currently **1.99.0**. They track the toolchain
  the project develops and qualifies against, not the advertised floor.

The dedicated `msrv` job in `.github/workflows/ci.yml` reads
`package.rust-version` from `Cargo.toml` and runs
`cargo +$MSRV check --locked --lib` (and `--features server`) on it, so the
advertised floor is exercised in CI independently of the dev pin.

Notes:

- The `zmq` feature builds vendored libzmq from C++ and is not covered by the
  `msrv` job; it stays in the `validate` job on the dev pin.
- `cargo test --all-targets` additionally requires dev-dependencies
  (`proptest` 1.12 needs rustc 1.88), so the test suite floor is higher than
  the library MSRV. Run tests on the dev pin (1.99.0), not on `rust-version`.
- This split follows the stack-wide policy in
  [Limen-Neural/neuromod#214](https://github.com/Limen-Neural/neuromod/issues/214).
