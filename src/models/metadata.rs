// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, check_count, check_opt_string, check_string,
};

use super::de::{de_metadata_custom, de_source};

/// Optional batch metadata for debugging and monitoring.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(try_from = "BatchMetadataWire")]
pub struct BatchMetadata {
    /// Processing latency in nanoseconds.
    pub processing_latency_ns: Option<u64>,
    /// Source identifier (for example, "encoder", "compute_layer_2").
    pub source: Option<String>,
    /// Additional metadata fields.
    pub custom: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
struct BatchMetadataWire {
    processing_latency_ns: Option<u64>,
    #[serde(default, deserialize_with = "de_source")]
    source: Option<String>,
    #[serde(default, deserialize_with = "de_metadata_custom")]
    custom: std::collections::HashMap<String, String>,
}

impl TryFrom<BatchMetadataWire> for BatchMetadata {
    type Error = ValidationError;

    fn try_from(wire: BatchMetadataWire) -> Result<Self, Self::Error> {
        let metadata = BatchMetadata {
            processing_latency_ns: wire.processing_latency_ns,
            source: wire.source,
            custom: wire.custom,
        };
        metadata.validate()?;
        Ok(metadata)
    }
}

impl Validate for BatchMetadata {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("source", self.source.as_deref(), limits)?;
        check_count("custom", self.custom.len(), limits.max_metadata_entries)?;
        for (key, value) in &self.custom {
            if key.is_empty() {
                return Err(ValidationError::nested_metadata("custom.<empty>"));
            }
            check_string("custom.<key>", key, limits)?;
            check_string("custom.<value>", value, limits)?;
        }
        Ok(())
    }
}
