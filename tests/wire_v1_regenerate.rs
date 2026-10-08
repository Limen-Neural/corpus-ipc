// SPDX-License-Identifier: MIT OR Apache-2.0

//! Intentional regeneration of the wire-v1 golden corpus (LIM-1446 / RM-1330).
//!
//! Gated on `REGENERATE_WIRE_V1=1` so it never runs accidentally in CI. When
//! enabled it rewrites the `positive` fixture files from
//! `encode_canonical_ipc_message` and recomputes every SHA-256 in
//! `manifest.json`. Hand-maintained fixture *content* is never touched.
//! Inspect the diff before committing — this is the only supported way to
//! change canonical bytes.

mod shared;

use std::collections::HashMap;
use std::path::Path;

use corpus_ipc::encode_canonical_ipc_message;
use shared::{
    ManifestEntry, REGENERATE_ENV, fixture_root, load_manifest, positive_vectors, sha256_hex,
};

const HEADER: &str = "{\n  \"schema_version\": 1,\n  \"encoding_profile\": \"corpus-ipc \
     wire v1 canonical JSON (docs/wire-encoding.md)\",\n  \"generator\": \"REGENERATE_WIRE_V1=1 \
     cargo test --locked --test wire_v1_regenerate\",\n  \"fixtures\": [\n";

#[test]
fn regenerate_fixtures() {
    if std::env::var_os(REGENERATE_ENV).is_none() {
        eprintln!("skipping regeneration: set {REGENERATE_ENV}=1 to enable");
        return;
    }
    let root = fixture_root();
    let manifest = load_manifest();
    let mut entries = Vec::new();
    for entry in &manifest.fixtures {
        let path = root.join(&entry.file);
        if entry.kind == "positive" {
            rewrite_positive(&path, entry);
        }
        entries.push(entry_json(&path, entry));
    }
    let mut out = String::from(HEADER);
    out.push_str(&entries.join(",\n"));
    out.push_str("\n  ]\n}\n");
    std::fs::write(root.join("manifest.json"), out).expect("write manifest.json");
    eprintln!(
        "regenerated {} fixtures under {}",
        entries.len(),
        root.display()
    );
}

fn rewrite_positive(path: &Path, entry: &ManifestEntry) {
    let messages: HashMap<&str, corpus_ipc::IpcMessage> = positive_vectors()
        .into_iter()
        .map(|(n, _, m)| (n, m))
        .collect();
    let message = messages
        .get(entry.name.as_str())
        .unwrap_or_else(|| panic!("no constructor for positive {}", entry.name));
    let bytes = encode_canonical_ipc_message(message)
        .unwrap_or_else(|e| panic!("encode {}: {e}", entry.name));
    std::fs::write(path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn entry_json(path: &Path, entry: &ManifestEntry) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let variant = entry
        .variant
        .as_ref()
        .map(|v| format!(",\n      \"variant\": \"{v}\""))
        .unwrap_or_default();
    format!(
        "    {{\n      \"name\": \"{}\",\n      \"file\": \"{}\",\n      \"sha256\": \
         \"{}\",\n      \"kind\": \"{}\"{},\n      \"expect\": \"{}\"\n    }}",
        entry.name,
        entry.file,
        sha256_hex(&bytes),
        entry.kind,
        variant,
        entry.expect
    )
}
