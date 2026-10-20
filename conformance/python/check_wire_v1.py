"""
Offline wire-v1 conformance harness (LIM-1327).

Reads test-vectors/wire-v1, checks each SHA-256, and applies the same
decode and rejection rules as decode_ipc_message_json.
"""

from __future__ import annotations

import sys

from wire_v1.cases import check_named_cases, check_numeric_contract
from wire_v1.corpus import check_fixture, check_manifest_hashes, load_manifest, vector_dir


def main() -> int:
    check_numeric_contract()
    root = vector_dir()
    fixtures = check_manifest_hashes(root, load_manifest(root))
    for entry in fixtures:
        check_fixture(root, entry)
    check_named_cases(root, fixtures)
    print(f"ok {len(fixtures)} wire-v1 fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
