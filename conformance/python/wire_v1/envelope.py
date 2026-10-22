"""Version classification and externally tagged IpcMessage decode."""

from __future__ import annotations

import json

from wire_v1.decode_input import decode_embeddings, decode_neuromodulators, decode_spikes, decode_stimuli
from wire_v1.decode_output import decode_config, decode_gradients, decode_loss, decode_traces
from wire_v1.errors import WireError, payload_error
from wire_v1.limits import CURRENT, LEGACY_UNVERSIONED, MIN_SUPPORTED, UNIT_VARIANTS, classify_wire_version
from wire_v1.model import Message

DECODERS = {
    "Spikes": decode_spikes,
    "Embeddings": decode_embeddings,
    "Stimuli": decode_stimuli,
    "Neuromodulators": decode_neuromodulators,
    "Loss": decode_loss,
    "ConfigUpdate": decode_config,
    "GradientUpdate": decode_gradients,
    "EligibilityTraces": decode_traces,
}


def decode_message(data: bytes) -> Message:
    value = _parse_json(data)
    if _is_envelope(value):
        return _decode_envelope(value)
    _accept_legacy()
    return decode_payload(value)


def _is_envelope(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return "wire_version" in value


def _parse_json(data: bytes) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WireError("json_error", f"envelope JSON error: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireError("json_error", f"envelope JSON error: {exc}") from exc


def _decode_envelope(value: dict) -> Message:
    version = _parse_version(value["wire_version"])
    _reject_unsupported(version)
    if "payload" not in value:
        raise WireError("missing_payload", "missing payload on compatibility envelope")
    return decode_payload(value["payload"])


def _reject_unsupported(version: int) -> None:
    kind = classify_wire_version(version)
    if kind == "too_old":
        raise WireError("too_old", _too_old_reason(version))
    if kind == "too_new":
        raise WireError("too_new", _too_new_reason(version))


def _too_old_reason(version: int) -> str:
    return (
        f"wire version {version} is too old (minimum supported is {MIN_SUPPORTED}); "
        "payload was not interpreted"
    )


def _too_new_reason(version: int) -> str:
    return (
        f"wire version {version} is too new (current is {CURRENT}); "
        "payload was not interpreted"
    )


def _parse_version(value: object) -> int:
    if _not_u32_token(value):
        raise _bad_version(value)
    if value < 0:
        raise _bad_version(value)
    if value > 2**32 - 1:
        raise _bad_version(value)
    return value


def _not_u32_token(value: object) -> bool:
    if isinstance(value, bool):
        return True
    return not isinstance(value, int)


def _bad_version(value: object) -> WireError:
    return WireError(
        "invalid_version",
        f"wire_version must be a non-negative integer fitting u32, got {value!r}",
    )


def _accept_legacy() -> None:
    if classify_wire_version(LEGACY_UNVERSIONED) != "supported":
        raise WireError("too_old", "legacy unversioned JSON is outside the supported window")


def decode_payload(value: object) -> Message:
    if isinstance(value, str):
        return _unit_string(value)
    if not isinstance(value, dict):
        raise _not_object()
    _reject_internal_tag(value)
    return _tagged_object(value)


def _unit_string(value: str) -> Message:
    if value in UNIT_VARIANTS:
        return Message(value, None)
    raise payload_error(f"unknown IpcMessage variant {value!r}")


def _not_object() -> WireError:
    return WireError(
        "not_an_object",
        "IpcMessage must be an externally tagged object or a unit-variant string",
    )


def _reject_internal_tag(value: dict) -> None:
    if "type" not in value:
        return
    if set(value) == {"type"}:
        return
    raise payload_error(
        "internally tagged message is not supported "
        f"(found type={value.get('type')!r}); expected an externally tagged IpcMessage"
    )


def _tagged_object(value: dict) -> Message:
    if len(value) != 1:
        found = sorted(value)
        raise payload_error(
            "externally tagged IpcMessage must have exactly one variant key, "
            f"found {found}"
        )
    name, body = next(iter(value.items()))
    return _decode_variant(name, body)


def _decode_variant(name: str, body: object) -> Message:
    if name in UNIT_VARIANTS:
        return _unit_object(name, body)
    decoder = DECODERS.get(name)
    if decoder is None:
        raise payload_error(f"unknown IpcMessage variant {name!r}")
    return Message(name, decoder(body))


def _unit_object(name: str, body: object) -> Message:
    if body not in (None, {}):
        raise payload_error(f"{name} is a unit variant and must be a JSON string")
    return Message(name, None)
