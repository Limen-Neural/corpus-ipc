"""
Batch metadata and the optional metadata field on spike and stimulus batches.
"""

from __future__ import annotations

from wire_v1.errors import payload_error
from wire_v1.limits import MAX_METADATA_ENTRIES
from wire_v1.model import BatchMetadata
from wire_v1.scalars import (
    add_total,
    as_u64,
    check_count,
    opt_session,
    require_object,
    string_bytes,
)


def read_optional_metadata(obj: dict) -> BatchMetadata | None:
    """Missing metadata and JSON null are both None, matching serde's Option rule."""
    if "metadata" not in obj:
        return None
    if obj["metadata"] is None:
        return None
    return decode_metadata(obj["metadata"], "metadata")


def decode_metadata(value: object, path: str) -> BatchMetadata:
    obj = require_object(value, path)
    metadata = BatchMetadata(read_latency(obj, path), read_source(obj, path), read_custom(obj, path))
    check_count(len(metadata.custom), MAX_METADATA_ENTRIES, f"{path}.custom")
    return metadata


def read_latency(obj: dict, path: str) -> int | None:
    """A missing processing_latency_ns is None. Null is also None."""
    if "processing_latency_ns" not in obj:
        return None
    latency = obj["processing_latency_ns"]
    if latency is None:
        return None
    return as_u64(latency, f"{path}.processing_latency_ns")


def read_source(obj: dict, path: str) -> str | None:
    if "source" not in obj:
        return None
    return opt_session(obj["source"], f"{path}.source")


def read_custom(obj: dict, path: str) -> dict[str, str]:
    if "custom" not in obj:
        return {}
    return _custom_map(obj["custom"], path)


def _custom_map(value: object, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise_custom_type(path)
    check_count(len(value), MAX_METADATA_ENTRIES, f"{path}.custom")
    custom: dict[str, str] = {}
    for key, item in value.items():
        custom[key] = _custom_entry(key, item, path)
    return custom


def raise_custom_type(path: str) -> None:
    raise payload_error(f"{path}.custom must be an object")


def _custom_entry(key: str, item: object, path: str) -> str:
    if key == "":
        raise payload_error(f"{path}.custom.<empty> is not a valid key")
    string_bytes(key, f"{path}.custom.<key>")
    if not isinstance(item, str):
        raise payload_error(f"{path}.custom.{key} must be a string")
    string_bytes(item, f"{path}.custom.<value>")
    return item


def validate_metadata(metadata: BatchMetadata | None, already: int, path: str) -> None:
    total = add_total(0, already, path)
    if metadata is None:
        return
    _recheck_metadata(metadata, path)
    add_total(total, len(metadata.custom), path)


def _recheck_metadata(metadata: BatchMetadata, path: str) -> None:
    if metadata.source is not None:
        string_bytes(metadata.source, f"{path}.source")
    check_count(len(metadata.custom), MAX_METADATA_ENTRIES, f"{path}.custom")
    for key, item in metadata.custom.items():
        _custom_entry(key, item, path)
