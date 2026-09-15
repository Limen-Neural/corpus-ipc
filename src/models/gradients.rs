// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, check_count, check_finite_slice,
    check_string, check_unique_strings,
};

use super::de::{
    de_eligibility_trace, de_gradient_rows, de_gradient_values, de_layer_id, de_session_id,
    prefix_path,
};

/// Gradient update batch from external training or optimization algorithms.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "GradientBatchWire")]
pub struct GradientBatch {
    /// Session ID for routing back to correct experiment.
    pub session_id: String,
    /// Batch ID correlation with original input.
    pub batch_id: u64,
    /// Individual gradient updates.
    pub gradients: Vec<GradientUpdate>,
}

#[derive(Deserialize)]
struct GradientBatchWire {
    #[serde(deserialize_with = "de_session_id")]
    session_id: String,
    batch_id: u64,
    #[serde(deserialize_with = "de_gradient_rows")]
    gradients: Vec<GradientUpdate>,
}

impl TryFrom<GradientBatchWire> for GradientBatch {
    type Error = ValidationError;

    fn try_from(wire: GradientBatchWire) -> Result<Self, Self::Error> {
        let batch = GradientBatch {
            session_id: wire.session_id,
            batch_id: wire.batch_id,
            gradients: wire.gradients,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Validate for GradientBatch {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("session_id", &self.session_id, limits)?;
        check_count("gradients", self.gradients.len(), limits.max_gradients)?;
        let mut total = 0;
        add_to_total(&mut total, self.gradients.len(), limits, "aggregate")?;
        accumulate_gradient_rows(&self.gradients, limits, &mut total)?;
        check_unique_strings("gradients", &self.gradients, |row| row.layer_id.as_str())?;
        Ok(())
    }
}

fn accumulate_gradient_rows(
    rows: &[GradientUpdate],
    limits: ProtocolLimits,
    total: &mut usize,
) -> Result<(), ValidationError> {
    for (index, update) in rows.iter().enumerate() {
        update
            .validate_with(limits)
            .map_err(|err| prefix_path(err, &format!("gradients[{index}]")))?;
        add_to_total(total, update.gradients.len(), limits, "aggregate")?;
        if let Some(trace) = &update.eligibility_trace {
            add_to_total(total, trace.len(), limits, "aggregate")?;
        }
    }
    Ok(())
}

/// Individual gradient update for a specific layer or parameter.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "GradientUpdateWire")]
pub struct GradientUpdate {
    /// Target layer identifier.
    pub layer_id: String,
    /// Gradient values (flattened or sparse representation).
    pub gradients: Vec<f32>,
    /// Optional eligibility trace for E-prop algorithms.
    pub eligibility_trace: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct GradientUpdateWire {
    #[serde(deserialize_with = "de_layer_id")]
    layer_id: String,
    #[serde(deserialize_with = "de_gradient_values")]
    gradients: Vec<f32>,
    #[serde(default, deserialize_with = "de_eligibility_trace")]
    eligibility_trace: Option<Vec<f32>>,
}

impl TryFrom<GradientUpdateWire> for GradientUpdate {
    type Error = ValidationError;

    fn try_from(wire: GradientUpdateWire) -> Result<Self, Self::Error> {
        let update = GradientUpdate {
            layer_id: wire.layer_id,
            gradients: wire.gradients,
            eligibility_trace: wire.eligibility_trace,
        };
        update.validate()?;
        Ok(update)
    }
}

impl Validate for GradientUpdate {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_string("layer_id", &self.layer_id, limits)?;
        check_count("gradients", self.gradients.len(), limits.max_channel_values)?;
        check_finite_slice("gradients", &self.gradients)?;
        if let Some(trace) = &self.eligibility_trace {
            check_count("eligibility_trace", trace.len(), limits.max_channel_values)?;
            check_finite_slice("eligibility_trace", trace)?;
        }
        Ok(())
    }
}
