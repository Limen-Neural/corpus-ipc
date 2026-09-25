# ZMQ readout golden frames (LIM-1275 / LIM-1328)

Little-endian binary fixtures for the production [`parse_readout_packet`](../../src/zmq_readout.rs) path.

## Encoding

- Each `*.hex` file holds **lowercase hex** of the raw packet bytes (no spaces).
- `manifest.json` lists SHA-256 digests of the **decoded** bytes (not the hex text).
- Repo-only: excluded from the crates.io `include` allow-list; integration tests load via `CARGO_MANIFEST_DIR`.

## Frame semantics

| Fixture | Meaning |
| --- | --- |
| `empty_8_byte` | Tick header only (`N = 0` floats) |
| `one_float_12_byte` | Tick + 1×`f32` |
| `sixteen_float_72_byte` | Tick + 16×`f32` (typical lobe readout size) |
| `twenty_float_88_byte_historical_ambiguity` | Tick + 20×`f32`; same length as historical 16+4 modulator layout — **not** auto-split |
| `truncated_5_byte` / `misaligned_9_byte` | Rejected framing |
| `readout_at_max_floats` / `readout_max_plus_one_float` | Bounds at default cap (`1024` floats) |

Verify digests:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("test-vectors/zmq")
manifest = json.loads((root / "manifest.json").read_text())
for entry in manifest["fixtures"]:
    data = bytes.fromhex((root / entry["file"]).read_text().strip())
    digest = hashlib.sha256(data).hexdigest()
    assert digest == entry["sha256"], entry["name"]
print("ok", len(manifest["fixtures"]))
PY
```
