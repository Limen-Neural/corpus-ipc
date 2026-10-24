#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# C++ driver for the vendored libzmq 4.3.4 (zeromq-src 0.2.6, via zmq-sys).
# Compiles every translation unit with ${CORPUS_IPC_REAL_CXX:-g++}, and
# rewrites vendor/src/pipe.cpp first.
#
# libzmq 4.3.4 and 4.3.5 double-free ypipe storage in
# pipe_t::process_pipe_term_ack when shutdown commands overlap on the I/O
# thread (zeromq/libzmq#4806). That is a SIGSEGV, and Ubuntu 26.04 GitHub
# runners hit it in `cargo test --all-features`. Upstream fixed it after
# 4.3.5 in acef50b8b21bd0e18851f891453e085d993a30aa (PR 4853): null _in_pipe
# before draining and deleting it. crates.io still ships 4.3.4, so this
# wrapper applies that change to the one file while it is compiled.
# Every other argument is forwarded unchanged. If pipe.cpp no longer
# contains the expected body, the driver exits non-zero instead of
# compiling the unfixed source.

set -euo pipefail

real="${CORPUS_IPC_REAL_CXX:-g++}"
args=()

for arg in "$@"; do
    case "$arg" in
        *zeromq-src*/vendor/src/pipe.cpp)
            tmp="$(mktemp "${TMPDIR:-/tmp}/corpus-ipc-pipe.XXXXXX")"
            mv "$tmp" "${tmp}.cpp"
            tmp="${tmp}.cpp"
            python3 - "$arg" "$tmp" <<'PY'
import pathlib
import sys

source_path, dest_path = sys.argv[1], sys.argv[2]
source = pathlib.Path(source_path).read_text()
old = """    if (!_conflate) {
        msg_t msg;
        while (_in_pipe->read (&msg)) {
            const int rc = msg.close ();
            errno_assert (rc == 0);
        }
    }

    LIBZMQ_DELETE (_in_pipe);"""
new = """    // corpus-ipc: null _in_pipe before delete (libzmq acef50b8, #4806)
    upipe_t *in_pipe = _in_pipe;
    _in_pipe = NULL;

    if (!_conflate && in_pipe) {
        msg_t msg;
        while (in_pipe->read (&msg)) {
            const int rc = msg.close ();
            errno_assert (rc == 0);
        }
    }

    LIBZMQ_DELETE (in_pipe);"""
marker = "corpus-ipc: null _in_pipe before delete"
if old not in source:
    if marker in source:
        pathlib.Path(dest_path).write_text(source)
        sys.exit(0)
    sys.stderr.write(
        "cxx-libzmq-pipefix: pipe.cpp no longer matches the libzmq 4.3.4 "
        "process_pipe_term_ack body; refusing to compile it unfixed\n"
    )
    sys.exit(1)
if source.count(old) != 1:
    sys.stderr.write(
        "cxx-libzmq-pipefix: expected one process_pipe_term_ack drain site\n"
    )
    sys.exit(1)
pathlib.Path(dest_path).write_text(source.replace(old, new, 1))
PY
            args+=("$tmp")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

exec "$real" "${args[@]}"
