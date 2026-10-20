"""
Named wire-v1 cases that the manifest kinds do not state by themselves.
"""

from __future__ import annotations

import math
from pathlib import Path

from wire_v1.encode import encode_canonical
from wire_v1.envelope import decode_message
from wire_v1.errors import WireError, fail
from wire_v1.limits import classify_wire_version
from wire_v1.model import ConfigPayload, SpikeBatch, StimulusBatch, TraceBatch
from wire_v1.numbers import f64_to_f32, format_f64, uint_to_f32


def check_named_cases(root: Path, fixtures: list[dict]) -> None:
    by_name = {entry["name"]: entry for entry in fixtures}
    for check in NAMED_CHECKS:
        check(root, by_name)


def check_versions(_root: Path, _by_name: dict) -> None:
    if classify_wire_version(0) != "too_old":
        fail("classify", "version 0 must be too_old")
    if classify_wire_version(1) != "supported":
        fail("classify", "version 1 must be supported")
    if classify_wire_version(2) != "too_new":
        fail("classify", "version 2 must be too_new")


def check_too_new_before_payload(root: Path, by_name: dict) -> None:
    err = decode_error(root, by_name["v2_too_new_malformed_payload"])
    if err.kind != "too_new":
        fail("v2_too_new_malformed_payload", f"version must win, got {err}")
    if "payload was not interpreted" not in err.reason:
        fail("v2_too_new_malformed_payload", f"version must win, got {err}")
    if "2" not in err.reason:
        fail("v2_too_new_malformed_payload", "reason must name the found version")


def check_unknown_variant(root: Path, by_name: dict) -> None:
    err = decode_error(root, by_name["v1_unknown_variant"])
    require_reason(err, "payload_error", "NotARealVariant", "v1_unknown_variant")


def check_wrong_tag(root: Path, by_name: dict) -> None:
    err = decode_error(root, by_name["v1_wrong_tagging"])
    require_reason(err, "payload_error", "internally tagged", "v1_wrong_tagging")


def check_internal_stimuli(_root: Path, _by_name: dict) -> None:
    raw = b'{"wire_version":1,"payload":{"type":"Stimuli","values":[0.0],"valid_mask":null}}'
    err = expect_error(raw, "internal_stimuli")
    require_reason(err, "payload_error", "internally tagged", "internal_stimuli")


def check_alias(root: Path, by_name: dict) -> None:
    blob = fixture_bytes(root, by_name["alias_neuron_id_traces"])
    if b"neuron_id" not in blob:
        fail("alias_neuron_id_traces", "fixture must use the neuron_id alias only")
    if b"channel_id" in blob:
        fail("alias_neuron_id_traces", "fixture must use the neuron_id alias only")
    message = decode_message(blob)
    _require_alias_channel(message)
    canonical = encode_canonical(message)
    if b"neuron_id" in canonical:
        fail("alias_neuron_id_traces", "canonical output must use channel_id")
    if b'"channel_id":5' not in canonical:
        fail("alias_neuron_id_traces", "canonical output must use channel_id")


def _require_alias_channel(message) -> None:
    if not isinstance(message.body, TraceBatch):
        fail("alias_neuron_id_traces", "neuron_id did not decode as channel_id 5")
    if message.body.traces[0].channel_id != 5:
        fail("alias_neuron_id_traces", "neuron_id did not decode as channel_id 5")


def check_normalization(root: Path, by_name: dict) -> None:
    blob = fixture_bytes(root, by_name["normalization_integer_config"])
    if b"16777217" not in blob:
        fail("normalization_integer_config", "fixture must keep the literal 2^24+1 input")
    message = decode_message(blob)
    _require_normalized_floats(message)
    text = encode_canonical(message).decode()
    if "16777217" in text:
        fail("normalization_integer_config", f"canonical floats drifted: {text}")
    if '"big":16777216.0' not in text:
        fail("normalization_integer_config", f"canonical floats drifted: {text}")
    if '"n":42.0' not in text:
        fail("normalization_integer_config", f"42 must be a float: {text}")


def _require_normalized_floats(message) -> None:
    if not isinstance(message.body, ConfigPayload):
        fail("normalization_integer_config", "expected ConfigUpdate")
    big = message.body.config["big"]
    small = message.body.config["n"]
    if big.kind != "float":
        fail("normalization_integer_config", "JSON integers must decode as floats")
    if small.kind != "float":
        fail("normalization_integer_config", "JSON integers must decode as floats")
    if big.value != 16_777_216.0:
        fail("normalization_integer_config", f"big decoded as {big.value!r}")
    if small.value != 42.0:
        fail("normalization_integer_config", f"n decoded as {small.value!r}")


def check_signed_zero(root: Path, by_name: dict) -> None:
    message = decode_message(fixture_bytes(root, by_name["signed_zero"]))
    if not isinstance(message.body, ConfigPayload):
        fail("signed_zero", "expected ConfigUpdate")
    neg = message.body.config["neg"].value
    pos = message.body.config["pos"].value
    if math.copysign(1.0, neg) != -1.0:
        fail("signed_zero", f"signs were {neg!r} {pos!r}")
    if math.copysign(1.0, pos) != 1.0:
        fail("signed_zero", f"signs were {pos!r}")


def check_masks(root: Path, by_name: dict) -> None:
    masked = decode_message(fixture_bytes(root, by_name["stimuli_masked"]))
    unmasked = decode_message(fixture_bytes(root, by_name["stimuli_unmasked"]))
    _require_mask(masked)
    _require_unmasked(unmasked)
    err = decode_error(root, by_name["v1_invalid_stimulus_mask"])
    require_reason(err, "payload_error", "valid_mask", "v1_invalid_stimulus_mask")


def _require_mask(message) -> None:
    if not isinstance(message.body, StimulusBatch):
        fail("stimuli_masked", "valid_mask did not round-trip")
    if message.body.valid_mask != [True, False, True]:
        fail("stimuli_masked", "valid_mask did not round-trip")
    if message.body.values[1] != 0.0:
        fail("stimuli_masked", "masked placeholder value changed")


def _require_unmasked(message) -> None:
    if not isinstance(message.body, StimulusBatch):
        fail("stimuli_unmasked", "null valid_mask must decode as absent")
    if message.body.valid_mask is not None:
        fail("stimuli_unmasked", "null valid_mask must decode as absent")


def check_neuromod_range(root: Path, by_name: dict) -> None:
    err = decode_error(root, by_name["v1_invalid_neuromod_range"])
    require_reason(err, "payload_error", "dopamine", "v1_invalid_neuromod_range")


def check_legacy_ping(root: Path, by_name: dict) -> None:
    message = decode_message(fixture_bytes(root, by_name["legacy_unversioned_ping"]))
    if message.variant != "Ping":
        fail("legacy_unversioned_ping", "expected Ping")
    if encode_canonical(message) != fixture_bytes(root, by_name["ping"]):
        fail("legacy_unversioned_ping", "normalized Ping must match canonical ping.json")


def check_legacy_spikes(root: Path, by_name: dict) -> None:
    message = decode_message(fixture_bytes(root, by_name["legacy_unversioned_spikes"]))
    if not isinstance(message.body, SpikeBatch):
        fail("legacy_unversioned_spikes", "expected Spikes")
    _require_legacy_spike_fields(message.body)
    if not encode_canonical(message).startswith(b'{"payload":'):
        fail("legacy_unversioned_spikes", "normalized form must be a v1 envelope")


def _require_legacy_spike_fields(batch: SpikeBatch) -> None:
    if batch.session_id != "sess-1":
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    if batch.batch_id != 7:
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    if batch.metadata is not None:
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    if len(batch.spikes) != 1:
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    if batch.spikes[0].channel != 3:
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")


def check_missing_optionals(_root: Path, _by_name: dict) -> None:
    spikes = decode_message(b'{"Spikes":{"batch_id":1,"timestamp":2,"spikes":[]}}')
    _require_absent_metadata(spikes, "missing_spike_metadata")
    partial = (
        b'{"wire_version":1,"payload":{"Spikes":{"batch_id":1,"timestamp":2,"spikes":[],'
        b'"metadata":{"source":"encoder"}}}}'
    )
    _require_absent_latency(decode_message(partial))
    stimuli = (
        b'{"wire_version":1,"payload":{"Stimuli":{"session_id":null,"batch_id":1,'
        b'"timestamp":2,"values":[0.0],"valid_mask":null}}}'
    )
    _require_absent_metadata(decode_message(stimuli), "missing_stimulus_metadata")
    missing_mask = (
        b'{"wire_version":1,"payload":{"Stimuli":{"batch_id":1,"timestamp":2,'
        b'"values":[0.0],"session_id":null}}}'
    )
    err = expect_error(missing_mask, "missing_valid_mask")
    require_reason(err, "payload_error", "valid_mask", "missing_valid_mask")


def _require_absent_metadata(message, name: str) -> None:
    body = message.body
    if not isinstance(body, (SpikeBatch, StimulusBatch)):
        fail(name, "missing metadata must decode as null")
    if body.metadata is not None:
        fail(name, "missing metadata must decode as null")
    if b'"metadata":null' not in encode_canonical(message):
        fail(name, "canonical output must include metadata null")


def _require_absent_latency(message) -> None:
    if not isinstance(message.body, SpikeBatch):
        fail("missing_latency", "partial metadata must still decode")
    if message.body.metadata is None:
        fail("missing_latency", "partial metadata must still decode")
    if message.body.metadata.processing_latency_ns is not None:
        fail("missing_latency", "missing processing_latency_ns must be null")
    if message.body.metadata.source != "encoder":
        fail("missing_latency", "source must still decode")


def check_numeric_contract() -> None:
    if uint_to_f32(16_777_217) != 16_777_216.0:
        raise SystemExit("16777217 must round to f32 16777216")
    for value, text in _numeric_samples():
        got = format_f64(value)
        if got != text:
            raise SystemExit(f"float format {value!r}: got {got}, want {text}")


def _numeric_samples() -> list[tuple[float, str]]:
    return [
        (1e-6, "1e-6"),
        (1e-5, "0.00001"),
        (1e-4, "0.0001"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (16_777_216.0, "16777216.0"),
        (42.0, "42.0"),
        (-0.0, "-0.0"),
        (f64_to_f32(3.4028234663852886e38), "3.4028234663852886e+38"),
        (f64_to_f32(0.8), "0.800000011920929"),
        (f64_to_f32(1e-6), "9.999999974752427e-7"),
        (f64_to_f32(1e-5), "9.999999747378752e-6"),
        (f64_to_f32(1e20), "1.0000000200408773e+20"),
    ]


def require_reason(err: WireError, kind: str, needle: str, name: str) -> None:
    if err.kind != kind:
        fail(name, f"unexpected rejection {err}")
    if needle not in err.reason:
        fail(name, f"unexpected rejection {err}")


def fixture_bytes(root: Path, entry: dict) -> bytes:
    return (root / entry["file"]).read_bytes()


def decode_error(root: Path, entry: dict) -> WireError:
    return expect_error(fixture_bytes(root, entry), entry["name"])


def expect_error(data: bytes, name: str) -> WireError:
    try:
        decode_message(data)
    except WireError as exc:
        return exc
    fail(name, "expected a rejection")
    raise AssertionError("unreachable")


NAMED_CHECKS = (
    check_versions,
    check_too_new_before_payload,
    check_unknown_variant,
    check_wrong_tag,
    check_internal_stimuli,
    check_alias,
    check_normalization,
    check_signed_zero,
    check_masks,
    check_neuromod_range,
    check_legacy_ping,
    check_legacy_spikes,
    check_missing_optionals,
)
