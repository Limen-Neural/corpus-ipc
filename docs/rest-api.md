<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# REST API contract (v0.2)

`corpus_ipc_server` (crate feature `server`, Axum 0.8) exposes one
[`IpcBackend`](../src/trait_def.rs) instance over HTTP. This document is the
normative contract for the v0.2 REST surface; it ships in the crates.io
package alongside `docs/wire-encoding.md`.

The service binds `127.0.0.1:8080` by default (`CORPUS_IPC_BIND` overrides).

> **Boundary warning:** the router has no authentication or TLS. A remote
> bind (for example `0.0.0.0:8080`) must sit behind an external
> access-control boundary (network ACL, reverse proxy with auth/TLS, or
> equivalent). Do not expose it directly.

## Endpoint table

| Endpoint | Request body (`application/json`) | Success (200) body | Error statuses |
| --- | --- | --- | --- |
| `POST /initialize` | `{"model_path": string?}` — all fields optional, `{}` accepted | `{"ok": true, "message": "initialized"}` | 400, 415, 422, 502, 500 |
| `POST /process` | `{"inputs": [f32, ...]}` — required | `{"ok": true, "output": [f32, ...]}` | 400, 409, 415, 422, 502, 500 |
| `POST /save_state` | `{"model_path": string}` — required | `{"ok": true, "message": "state saved"}` | 400, 409, 415, 422, 502, 500 |
| `POST /reset` | none | `{"ok": true, "message": "reset"}` | 502, 500 |

Any method other than `POST` returns 405. A request to a path not listed
here returns 404.

`save_state` behavior is backend-specific: with the current backends it is a
successful no-op, not guaranteed persistence.

## Error envelope

Every non-2xx JSON error body uses one stable shape:

```json
{"ok": false, "error": {"code": "<code>", "message": "<human-readable detail>"}}
```

`code` is the stable machine-readable discriminator; `message` is
diagnostic text that may change between releases.

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 Bad Request | `bad_request` | Malformed JSON or unreadable request body |
| 415 Unsupported Media Type | `unsupported_media_type` | Missing or non-JSON `Content-Type` on a body endpoint |
| 422 Unprocessable Entity | `invalid_input` | Well-formed JSON that fails schema validation (missing/wrong-typed fields) or `BackendError::InvalidInput` |
| 409 Conflict | `not_initialized` | `/process` or `/save_state` before a successful `/initialize`, or after `/reset` |
| 502 Bad Gateway | `backend_error` | Backend initialization, processing, model-I/O, or communication failure |
| 500 Internal Server Error | `internal_error` | Service-internal failure (e.g. poisoned backend lock) |

## Lifecycle and reset semantics

- `/process` and `/save_state` require a live session: call `/initialize`
  first, otherwise the server returns 409 `not_initialized`.
- `/reset` is idempotent and always allowed. Both shipped backends clear
  their initialized state on reset, so the **next `/process` or
  `/save_state` returns 409 until `/initialize` is called again**. This is
  required for connection-oriented backends (e.g. ZMQ), which drop their
  transport socket on reset and must reconnect.
- `/initialize` is idempotent on success and may be called at any time.

## Migration from the 0.1 REST surface

The 0.1 binary returned HTTP 200 for every application-level result and used
`{"ok": false, "message": ...}` for backend errors; process-before-initialize
surfaced as `ok: false` inside a 200 body. v0.2 changes this deliberately:

- Success bodies gain `"ok": true`; `/process` returns `{"ok": true, "output": [...]}`.
- Errors now carry real HTTP statuses and the
  `{"ok": false, "error": {"code", "message"}}` envelope instead of
  `{"ok": false, "message"}`. Clients polling `ok` still detect failure, but
  clients that treated HTTP 200 as success must now consult the `error.code`
  classes above.
- `POST /process` bodies are unchanged (`{"inputs": [...]}`); response shape
  gains only the `ok` field.
