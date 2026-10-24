#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# C++ driver for the vendored libzmq 4.3.4 (zeromq-src 0.2.6, via zmq-sys).
# Compiles every translation unit with ${CORPUS_IPC_REAL_CXX:-g++}.
#
# Two source fixes, applied before the real compiler runs:
#
# 1. pipe.cpp: libzmq 4.3.4/4.3.5 double-frees ypipe storage in
#    pipe_t::process_pipe_term_ack (zeromq/libzmq#4806). Upstream fixed it
#    after 4.3.5 in acef50b8b21bd0e18851f891453e085d993a30aa. The driver
#    compiles a rewritten copy and exits non-zero if the expected body is
#    gone.
# 2. yqueue.hpp: command_t is 64-byte aligned, but zeromq-src's empty
#    platform.hpp leaves HAVE_POSIX_MEMALIGN unset, so queue chunks come
#    from malloc (16-byte alignment). On Ubuntu 26.04, cargo test
#    --all-features dies with SIGSEGV in ypipe_t<command_t,16>::write while
#    storing a command into that chunk (`_queue.back() = value_`), including
#    the first write to a socket mailbox during zmq_connect. The driver
#    allocates those chunks with posix_memalign(64) and, if the writer
#    cursor was cleared, restores it from the end cursor before that store.
#    Quoted includes resolve yqueue.hpp next to the including file, so the
#    vendored header is replaced in place. A mismatch exits non-zero.

set -euo pipefail

real="${CORPUS_IPC_REAL_CXX:-g++}"
# A previous export can point the "real" compiler back at this script.
if [[ "$(basename "$real")" == "$(basename "$0")" ]]; then
    real=g++
fi
args=()
yqueue=""

for arg in "$@"; do
    if [[ "$arg" == *zeromq-src*/vendor/src/*.cpp ]]; then
        yqueue="$(dirname "$arg")/yqueue.hpp"
        if [[ "$arg" != */pipe.cpp ]]; then
            args+=("$arg")
            continue
        fi
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
    else
        args+=("$arg")
    fi
done

if [[ -n "$yqueue" ]]; then
    exec 9>>"${yqueue}.lock"
    flock 9
    python3 - "$yqueue" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text()
marker = "corpus-ipc: 64-byte yqueue chunks"
if marker in source:
    sys.exit(0)

old_back = "    inline T &back () { return _back_chunk->values[_back_pos]; }"
new_back = """    // corpus-ipc: 64-byte yqueue chunks; restore a cleared writer cursor
    inline T &back ()
    {
        if (!_back_chunk && _end_chunk) {
            _back_chunk = _end_chunk;
            _back_pos = _end_pos ? _end_pos - 1 : 0;
        }
        return _back_chunk->values[_back_pos];
    }"""
old_alloc = """    static inline chunk_t *allocate_chunk ()
    {
#ifdef HAVE_POSIX_MEMALIGN
        void *pv;
        if (posix_memalign (&pv, ALIGN, sizeof (chunk_t)) == 0)
            return (chunk_t *) pv;
        return NULL;
#else
        return static_cast<chunk_t *> (malloc (sizeof (chunk_t)));
#endif
    }"""
new_alloc = """    // corpus-ipc: command_t is aligned(64); malloc chunks are not.
    static inline chunk_t *allocate_chunk ()
    {
        void *pv = NULL;
        if (posix_memalign (&pv, 64, sizeof (chunk_t)) != 0)
            return NULL;
        return static_cast<chunk_t *> (pv);
    }"""
if old_back not in source or old_alloc not in source:
    sys.stderr.write(
        "cxx-libzmq-pipefix: yqueue.hpp no longer matches libzmq 4.3.4; "
        "refusing to compile it unfixed\n"
    )
    sys.exit(1)
source = source.replace(old_back, new_back, 1).replace(old_alloc, new_alloc, 1)
tmp = path.with_suffix(".hpp.corpus-ipc")
tmp.write_text(source)
os.replace(tmp, path)
PY
    flock -u 9
    exec 9>&-
    args+=("-fno-strict-aliasing")
fi

exec "$real" "${args[@]}"
