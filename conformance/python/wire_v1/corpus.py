"""
Manifest integrity and per-fixture expectations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wire_v1.encode import encode_canonical
from wire_v1.envelope import decode_message
from wire_v1.errors import WireError, fail
from wire_v1.limits import DECODE_OK_KINDS, EXPECTATIONS, KNOWN_VARIANTS


def vector_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "test-vectors" / "wire-v1"


def load_manifest(root: Path) -> dict:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read manifest: {exc}") from exc
    _require_manifest_shape(manifest)
    return manifest


def _require_manifest_shape(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise SystemExit("manifest schema_version must be 1")
    if "wire v1" not in manifest.get("encoding_profile", ""):
        raise SystemExit("manifest encoding_profile must name wire v1")


def check_manifest_hashes(root: Path, manifest: dict) -> list[dict]:
    fixtures = manifest["fixtures"]
    seen: set[str] = set()
    for entry in fixtures:
        check_manifest_entry(root, entry, seen)
    require_variant_coverage(fixtures)
    return fixtures


def check_manifest_entry(root: Path, entry: dict, seen: set[str]) -> None:
    name = entry["name"]
    if name in seen:
        raise SystemExit(f"duplicate fixture name {name}")
    seen.add(name)
    require_expectation(entry)
    require_positive_variant(entry)
    require_digest(root, entry)


def require_expectation(entry: dict) -> None:
    if entry["expect"] not in EXPECTATIONS:
        raise SystemExit(f"{entry['name']}: unknown expectation {entry['expect']}")


def require_positive_variant(entry: dict) -> None:
    if entry["kind"] != "positive":
        return
    if entry.get("variant") not in KNOWN_VARIANTS:
        raise SystemExit(f"{entry['name']}: unknown positive variant {entry.get('variant')}")


def require_digest(root: Path, entry: dict) -> None:
    digest = hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        fail(entry["name"], f"sha256 drift: {digest} != {entry['sha256']}")


def require_variant_coverage(fixtures: list[dict]) -> None:
    covered = {entry["variant"] for entry in fixtures if entry["kind"] == "positive"}
    missing = [variant for variant in KNOWN_VARIANTS if variant not in covered]
    if missing:
        raise SystemExit(f"missing positive vector for {missing}")


def check_fixture(root: Path, entry: dict) -> None:
    blob = (root / entry["file"]).read_bytes()
    try:
        message = decode_message(blob)
    except WireError as exc:
        accept_rejection(entry, exc)
        return
    accept_success(entry, message, blob)


def accept_rejection(entry: dict, exc: WireError) -> None:
    name = entry["name"]
    if entry["expect"] == "decode_ok":
        fail(name, f"expected decode_ok, got {exc.kind}: {exc.reason}")
    if exc.kind != entry["expect"]:
        fail(name, f"expected {entry['expect']}, got {exc.kind}: {exc.reason}")
    if not exc.reason.strip():
        fail(name, "rejection reason is empty")


def accept_success(entry: dict, message, blob: bytes) -> None:
    if entry["expect"] != "decode_ok":
        fail(entry["name"], f"expected {entry['expect']}, but decode succeeded as {message.variant}")
    if entry["kind"] not in DECODE_OK_KINDS:
        fail(entry["name"], f"unexpected kind {entry['kind']}")
    if entry["kind"] == "positive":
        require_positive(entry, message, blob)
        return
    require_normalized(entry, message, blob)


def require_positive(entry: dict, message, blob: bytes) -> None:
    if message.variant != entry["variant"]:
        fail(entry["name"], f"variant {message.variant} != {entry['variant']}")
    encoded = encode_canonical(message)
    if encoded != blob:
        fail(entry["name"], "canonical re-encode drifted\n" f"  got {encoded.decode()}\n" f"  want {blob.decode()}")


def require_normalized(entry: dict, message, blob: bytes) -> None:
    encoded = encode_canonical(message)
    again = decode_message(encoded)
    if encode_canonical(again) != encoded:
        fail(entry["name"], "normalized re-encode is not stable")
    if encoded == blob:
        fail(entry["name"], "hand-maintained input was already canonical; check the fixture kind")
