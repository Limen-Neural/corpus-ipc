// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::Deserializer;

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, bounded_map, bounded_opt_string,
    bounded_opt_vec, bounded_string, bounded_vec, finite_f32_at,
};

use super::{BatchMetadata, ConfigValue, GradientUpdate, SpikeEvent, TraceData};

pub(super) fn de_opt_session_id<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Option<String>, D::Error> {
    bounded_opt_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "session_id")
}

pub(super) fn de_session_id<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "session_id")
}

pub(super) fn de_source<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    bounded_opt_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "source")
}

pub(super) fn de_layer_id<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "layer_id")
}

pub(super) fn de_values<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "values")
}

pub(super) fn de_embedding<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "embedding")
}

pub(super) fn de_valid_mask<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<bool>>, D::Error> {
    bounded_opt_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "valid_mask")
}

pub(super) fn de_spikes<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<SpikeEvent>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_spike_events, "spikes")
}

pub(super) fn de_traces<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<TraceData>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_traces, "traces")
}

pub(super) fn de_gradient_rows<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Vec<GradientUpdate>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_gradients, "gradients")
}

pub(super) fn de_gradient_values<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "gradients")
}

pub(super) fn de_eligibility_trace<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<f32>>, D::Error> {
    bounded_opt_vec(
        d,
        ProtocolLimits::DEFAULT.max_channel_values,
        "eligibility_trace",
    )
}

pub(super) fn de_float_array<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    bounded_vec(d, ProtocolLimits::DEFAULT.max_channel_values, "config")
}

pub(super) fn de_config_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    bounded_string(d, ProtocolLimits::DEFAULT.max_string_bytes, "config")
}

pub(super) fn de_metadata_custom<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<std::collections::HashMap<String, String>, D::Error> {
    bounded_map(
        d,
        ProtocolLimits::DEFAULT.max_metadata_entries,
        ProtocolLimits::DEFAULT.max_string_bytes,
        "custom",
    )
}

pub(super) fn de_config_map<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<std::collections::HashMap<String, ConfigValue>, D::Error> {
    bounded_map(
        d,
        ProtocolLimits::DEFAULT.max_metadata_entries,
        ProtocolLimits::DEFAULT.max_string_bytes,
        "config",
    )
}

pub(super) fn de_finite_f32<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "value")
}

pub(super) fn de_loss<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "Loss")
}

pub(super) fn de_strength<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "strength")
}

pub(super) fn de_trace_value<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    finite_f32_at(d, "trace_value")
}

pub(super) fn prefix_path(mut err: ValidationError, prefix: &str) -> ValidationError {
    if err.path.is_empty() {
        err.path = prefix.to_owned();
    } else {
        err.path = format!("{prefix}.{}", err.path);
    }
    err
}

pub(super) fn validate_optional_metadata(
    metadata: &Option<BatchMetadata>,
    limits: ProtocolLimits,
    already_counted: usize,
) -> Result<(), ValidationError> {
    let mut total = 0;
    add_to_total(&mut total, already_counted, limits, "aggregate")?;
    if let Some(metadata) = metadata {
        metadata.validate_with(limits)?;
        add_to_total(&mut total, metadata.custom.len(), limits, "aggregate")?;
    }
    Ok(())
}
