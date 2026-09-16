// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::validation::{
    ProtocolLimits, Validate, ValidationError, add_to_total, check_count, check_finite,
    check_finite_slice, check_opt_string, check_string,
};

use super::de::{
    de_config_map, de_config_string, de_finite_f32, de_float_array, de_opt_session_id, prefix_path,
};

/// Configuration payload for runtime parameter updates.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(try_from = "ConfigPayloadWire")]
pub struct ConfigPayload {
    /// Target session (None = global).
    pub session_id: Option<String>,
    /// Configuration key-value pairs.
    pub config: std::collections::HashMap<String, ConfigValue>,
}

#[derive(Deserialize)]
struct ConfigPayloadWire {
    #[serde(deserialize_with = "de_opt_session_id")]
    session_id: Option<String>,
    #[serde(deserialize_with = "de_config_map")]
    config: std::collections::HashMap<String, ConfigValue>,
}

impl TryFrom<ConfigPayloadWire> for ConfigPayload {
    type Error = ValidationError;

    fn try_from(wire: ConfigPayloadWire) -> Result<Self, Self::Error> {
        let payload = ConfigPayload {
            session_id: wire.session_id,
            config: wire.config,
        };
        payload.validate()?;
        Ok(payload)
    }
}

impl Validate for ConfigPayload {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        check_opt_string("session_id", self.session_id.as_deref(), limits)?;
        check_count("config", self.config.len(), limits.max_metadata_entries)?;
        let mut total = 0;
        add_to_total(&mut total, self.config.len(), limits, "aggregate")?;
        validate_config_entries(&self.config, limits, &mut total)
    }
}

fn validate_config_entries(
    config: &std::collections::HashMap<String, ConfigValue>,
    limits: ProtocolLimits,
    total: &mut usize,
) -> Result<(), ValidationError> {
    for (key, value) in config {
        if key.is_empty() {
            return Err(ValidationError::nested_metadata("config.<empty>"));
        }
        check_string("config.<key>", key, limits)?;
        value
            .validate_with(limits)
            .map_err(|err| prefix_path(err, &format!("config.{key}")))?;
        if let ConfigValue::FloatArray(values) = value {
            add_to_total(total, values.len(), limits, "aggregate")?;
        }
    }
    Ok(())
}

/// Configuration value types.
///
/// Uses `#[serde(untagged)]` so plain JSON numbers/strings/arrays/booleans
/// work directly inside `ConfigPayload::config`.
///
/// **Untagged deserialization behavior (intentional, pre-existing):**
/// `Float(f32)` is first, so JSON numbers (e.g. `42` or `1.5`) always
/// deserialize as `Float`. `Integer` is only reached for values that were
/// originally `ConfigValue::Integer` in Rust and then serialized, or under
/// certain deserializer configurations.
///
/// Round-tripping `Integer(42)` through JSON yields `Float(42.0)`.
/// Large integers (> ~2^24) may lose precision in f32.
/// Consumers relying on exact integer identity should be aware.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ConfigValue {
    /// Floating-point value.
    ///
    /// Because this is the first variant in an untagged enum, JSON
    /// numbers (integers and floats) deserialize as `Float`.
    Float(#[serde(deserialize_with = "de_finite_f32")] f32),

    /// Integer value (u64).
    ///
    /// Typically only produced when a Rust `ConfigValue::Integer` is
    /// serialized and round-tripped with the same serde configuration,
    /// or in specific deserializer contexts. Plain JSON numbers land
    /// in `Float` due to declaration order.
    Integer(u64),

    /// String value.
    ///
    /// Allows string-valued config (e.g. mode names, paths) in `ConfigPayload::config`.
    String(#[serde(deserialize_with = "de_config_string")] String),

    /// Boolean value.
    Boolean(bool),
    FloatArray(#[serde(deserialize_with = "de_float_array")] Vec<f32>),
}

impl Validate for ConfigValue {
    fn validate_with(&self, limits: ProtocolLimits) -> Result<(), ValidationError> {
        match self {
            Self::Float(value) => check_finite("value", *value),
            Self::Integer(_) | Self::Boolean(_) => Ok(()),
            Self::String(value) => check_string("value", value, limits),
            Self::FloatArray(values) => {
                check_count("value", values.len(), limits.max_channel_values)?;
                check_finite_slice("value", values)
            }
        }
    }
}
