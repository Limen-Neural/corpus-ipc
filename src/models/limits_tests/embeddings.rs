// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn embedding_round_trips() {
    let embedding = EmbeddingBatch {
        session_id: Some("sess".into()),
        batch_id: 1,
        embedding: vec![0.1, 0.2],
        sequence_length: 2,
    };
    let json = serde_json::to_value(&embedding).unwrap();
    let decoded: EmbeddingBatch = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, embedding);
    IpcMessage::Embeddings(embedding).validate().unwrap();
}

#[test]
fn embedding_rejects_oversize_sequence_length() {
    let embedding = EmbeddingBatch {
        session_id: Some("sess".into()),
        batch_id: 1,
        embedding: vec![0.1, 0.2],
        sequence_length: limits().max_aggregate_records + 1,
    };
    assert_eq!(
        embedding.validate().unwrap_err().kind,
        ValidationKind::LimitExceeded
    );
}
