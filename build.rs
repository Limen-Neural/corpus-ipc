// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ZMQ");

    if env::var_os("CARGO_FEATURE_ZMQ").is_none() {
        return;
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("src/zmq_api_lock.cpp");
    println!("cargo:rerun-if-changed={}", source.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out_dir.join("zmq_api_lock.o");
    let compiler = env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let status = Command::new(&compiler)
        .arg("-c")
        .arg("-O2")
        .arg("-fPIC")
        .arg("-pthread")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .unwrap_or_else(|err| panic!("failed to run {compiler} for zmq_api_lock.cpp: {err}"));
    if !status.success() {
        panic!("failed to compile {}", source.display());
    }

    println!("cargo:rustc-link-arg={}", object.display());
    for symbol in [
        "zmq_socket",
        "zmq_close",
        "zmq_connect",
        "zmq_bind",
        "zmq_unbind",
        "zmq_disconnect",
        "zmq_setsockopt",
        "zmq_getsockopt",
        "zmq_ctx_set",
    ] {
        println!("cargo:rustc-link-arg=-Wl,--wrap={symbol}");
    }
}
