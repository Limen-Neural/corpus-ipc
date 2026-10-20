"""Typed failures for the wire-v1 conformance harness."""


class WireError(Exception):
    """Decode or encode failure tagged with a manifest error class."""

    def __init__(self, kind: str, reason: str) -> None:
        """Store the error class and a human-readable reason."""
        super().__init__(f"{kind}: {reason}")
        self.kind = kind
        self.reason = reason


def payload_error(reason: str) -> WireError:
    return WireError("payload_error", reason)


def fail(name: str, detail: str) -> None:
    raise SystemExit(f"FAIL {name}: {detail}")
