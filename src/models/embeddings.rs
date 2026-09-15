// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, check_count, check_finite_slice,
    check_opt_string,
};

use super::de::{de_embedding, de_opt_session_id};

/// Batch of embeddings for projector and transformer components.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "EmbeddingBatchWire")]
pub struct EmbeddingBatch {
    /// Optional session ID for concurrent experiment isolation.
    pub session_id: Option<String>,
    /// Unique batch identifier for correlation.
    pub batch_id: u64,
    /// Embedding vector from compute processing.
    pub embedding: Vec<f32>,
    /// Sequence length for transformer compatibility.
    pub sequence_length: usize,
}

#[derive(Deserialize)]
struct EmbeddingBatchWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    batch_id: u64,
    #[serde(deserialize_with = "de_embedding")]
    embedding: Vec<f32>,
    sequence_length: usize,
}

impl TryFrom<EmbeddingBatchWire> for EmbeddingBatch {
    type Error = ValidationError;

    fn try_from(wire: EmbeddingBatchWire) -> Result<Self, Self::Error> {
        let batch = EmbeddingBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            embedding: wire.embedding,
            sequence_length: wire.sequence_length,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for EmbeddingBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("embedding", self.embedding.len(), limits.max_channel_values)?;
        check_count(
            "sequence_length",
            self.sequence_length,
            limits.max_aggregate_records,
        )?;
        check_finite_slice("embedding", &self.embedding)?;
        let mut total = 0;
        add_to_total(&mut total, self.embedding.len(), limits, "aggregate")?;
        Ok(())
    }
}
