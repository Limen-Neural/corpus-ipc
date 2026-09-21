<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# corpus-ipc wire encoding (profile v1)

This document specifies the **canonical wire encoding** produced by
`encode_canonical_ipc_message`. It is the single source of truth for
fixture generation and for downstream corpora (RM-1330, LIM-1331).

> This is the **corpus-ipc project wire profile**. It is **not** RFC 8785 /
> JCS and makes no JCS-compliance claim. It defines only what this crate emits
> and accepts.

## Envelope

Every canonical message is a wire-version-1 envelope:

```json
{"payload":<payload>,"wire_version":1}
```

(Keys are sorted, so `payload` precedes `wire_version` in the canonical form.)

- `wire_version` is always `1` (`WireCompatibility::CURRENT`).
- `payload` is an externally tagged `IpcMessage`:
  - struct variants encode as a single-key object, e.g. `{"Spikes":{...}}`;
  - unit variants encode as a JSON string, e.g. `"Ping"`, `"Shutdown"`.

Decoding goes through `decode_ipc_message_json`, which classifies the version
before the payload is used.

## Canonicalization rules

1. **Sorted keys, all levels.** Object member names are emitted in byte-wise
   ascending order at every nesting depth. This holds even though
   `ConfigPayload.config` and `BatchMetadata.custom` are `HashMap` in Rust:
   the encoder routes through `serde_json::Value` and calls
   `Value::sort_all_objects()` to recursively sort every nested object before
   serializing, so insertion order and per-process hash seed do not affect the
   bytes. The sort does not depend on serde_json's default `BTreeMap` backing —
   it stays correct even if a consumer's dependency graph enables the
   `preserve_order` feature (which switches `Value` to an insertion-ordered
   `IndexMap`).
2. **Compact.** No insignificant whitespace.
3. **UTF-8.** Output is UTF-8; non-ASCII keys and values are preserved.
4. **Determinism.** The same message encodes to identical bytes across runs
   and processes.

## Representable / normalized domain

The canonical encoder validates (`Validate::validate`) before serializing, so
only the representable domain reaches the wire:

- **Floats must be finite.** `NaN` and `±inf` are rejected with a typed
  `CanonicalEncodeError::Validation` (`ValidationKind::NonFinite`). They are
  never emitted as JSON `null`.
- **Signed zero is preserved.** `-0.0` and `0.0` are distinct IEEE-754 values
  and encode as `-0.0` and `0.0` respectively.
- **`ConfigValue` is Float-first (untagged).** JSON numbers decode as
  `ConfigValue::Float(f32)`. Consequently:
  - `ConfigValue::Integer(42)` round-trips through the wire as
    `ConfigValue::Float(42.0)`.
  - `f32` represents every integer up to `2^24` exactly; above `2^24`,
    integers can lose precision or identity on an `f32` round-trip (not every
    larger integer does, but some do — e.g. `16_777_217` is not representable).
    Consumers that need exact large-integer identity must not rely on
    `ConfigValue`.

  This Float-first behavior is intentional and unchanged; the canonical
  encoder does **not** redesign the numeric schema.

## Regenerating fixtures

Committed compatibility fixtures are produced from this one encoder. Do not
hand-edit fixture bytes; regenerate them from `encode_canonical_ipc_message`
so there is a single source of truth for the on-wire form.
