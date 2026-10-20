"""
Wire-v1 version window and default protocol limits.
"""

CURRENT = 1
MIN_SUPPORTED = 1
LEGACY_UNVERSIONED = 1

MAX_CHANNEL_VALUES = 65_536
MAX_SPIKE_EVENTS = 65_536
MAX_TRACES = 65_536
MAX_GRADIENTS = 65_536
MAX_METADATA_ENTRIES = 256
MAX_STRING_BYTES = 1_024
MAX_AGGREGATE = 131_072

UNIT_VARIANTS = ("Ping", "Shutdown", "TrainingComplete")
KNOWN_VARIANTS = (
    "Spikes",
    "Embeddings",
    "Stimuli",
    "Neuromodulators",
    "Loss",
    "ConfigUpdate",
    "GradientUpdate",
    "EligibilityTraces",
    *UNIT_VARIANTS,
)
EXPECTATIONS = ("decode_ok", "too_old", "too_new", "payload_error", "json_error")
DECODE_OK_KINDS = ("positive", "legacy", "alias", "normalization")


def classify_wire_version(version: int) -> str:
    if version < MIN_SUPPORTED:
        return "too_old"
    if version > CURRENT:
        return "too_new"
    return "supported"
