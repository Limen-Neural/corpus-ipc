<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Wire-v1 cross-language conformance

Offline Python and Julia readers of the golden corpus in
[`test-vectors/wire-v1/`](../test-vectors/wire-v1/README.md) (LIM-1327,
RM-1330 / LIM-1446). Both harnesses open those files, check the manifest
SHA-256 digests, and apply the wire-v1 rules in
[`docs/wire-encoding.md`](../docs/wire-encoding.md):

- version 1 envelopes and legacy externally tagged JSON, including unit
  variants such as `"Ping"`
- rejection of unknown `IpcMessage` variants, unsupported `wire_version`
  values (before the payload is interpreted), and internally tagged
  `{"type":"Stimuli",...}` messages
- `TraceData` input alias `neuron_id`, with canonical output using `channel_id`
- `f32` rounding (round-ties-to-even) and `ConfigValue` integer-to-float
  normalization, including `16777217` → `16777216.0`
- null versus missing fields, and `valid_mask` length

Positive fixtures must re-encode to the same canonical bytes. Legacy, alias,
and normalization fixtures must re-encode to that normalized domain, not to
a byte copy of their hand-written input. There is no second schema and no
separately generated corpus.

These harnesses do not build the Rust crate, do not use PyO3 or jlrs, and do
not open a ZeroMQ socket. `scripts/verify.sh` stays the Rust qualification
entry point and does not invoke them.

## Python

CPython 3.15 (current stable; the suite was run on 3.15.0, and CI installs the latest 3.15 patch). No third-party packages (`conformance/python/requirements.txt`).

```bash
python3 conformance/python/check_wire_v1.py
```

## Julia

Julia 1.13 (current stable; verified on 1.13.2). The only package is JSON.jl 0.21.4, pinned in
`conformance/julia/Project.toml` and `conformance/julia/Manifest.toml`.

```bash
julia --project=conformance/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=conformance/julia conformance/julia/check_wire_v1.jl
```

Linux CI runs both commands in the `wire-v1-conformance` job. That job does
not change the Rust feature or platform matrix.
