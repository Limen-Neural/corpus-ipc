// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use serde::de::{
    self, Deserializer, IntoDeserializer, MapAccess, SeqAccess, Visitor,
    value::{MapAccessDeserializer, SeqAccessDeserializer},
};
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
/// Serialize uses `#[serde(untagged)]` so plain JSON numbers/strings/arrays/booleans
/// are emitted inside `ConfigPayload::config`.
///
/// **JSON number behavior (intentional, pre-existing):**
/// JSON numbers (e.g. `42` or `1.5`) always deserialize as `Float`. `Integer`
/// is only reached for values that were originally `ConfigValue::Integer` in
/// Rust (in-memory), not from JSON. Deserialize is a `deserialize_any`
/// visitor rather than `#[serde(untagged)]` so JSON decimals still decode
/// when a consumer unifies serde_json `arbitrary_precision` (that feature
/// presents non-integer numbers to `deserialize_any` as a private Number
/// map, which untagged + `deserialize_with = f32` does not accept). The
/// `visit_map` arm accepts *only* that synthetic single-entry number map and
/// parses its decimal token directly to `f32`; any other object (including a
/// user object spoofing the magic key) is rejected.
///
/// Round-tripping `Integer(42)` through JSON yields `Float(42.0)`.
/// Large integers (> ~2^24) may lose precision in f32.
/// Consumers relying on exact integer identity should be aware.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ConfigValue {
    /// Floating-point value.
    ///
    /// JSON numbers (integers and floats) deserialize as `Float`.
    Float(f32),

    /// Integer value (u64).
    ///
    /// Typically only produced when a Rust `ConfigValue::Integer` is
    /// constructed in memory. Plain JSON numbers land in `Float`.
    Integer(u64),

    /// String value.
    ///
    /// Allows string-valued config (e.g. mode names, paths) in `ConfigPayload::config`.
    String(String),

    /// Boolean value.
    Boolean(bool),
    FloatArray(Vec<f32>),
}

impl<'de> Deserialize<'de> for ConfigValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ConfigValueVisitor;

        impl<'de> Visitor<'de> for ConfigValueVisitor {
            type Value = ConfigValue;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON number, string, boolean, or array of finite floats")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(ConfigValue::Boolean(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                de_finite_f32(value.into_deserializer()).map(ConfigValue::Float)
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                de_finite_f32(value.into_deserializer()).map(ConfigValue::Float)
            }

            fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
                de_finite_f32(value.into_deserializer()).map(ConfigValue::Float)
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                de_finite_f32(value.into_deserializer()).map(ConfigValue::Float)
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                de_config_string(value.into_deserializer()).map(ConfigValue::String)
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                de_config_string(value.into_deserializer()).map(ConfigValue::String)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                de_float_array(SeqAccessDeserializer::new(seq)).map(ConfigValue::FloatArray)
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                // serde_json `arbitrary_precision` presents JSON floats to
                // `deserialize_any` as a synthetic map keyed by the private token
                // `$serde_json::private::Number`. Only that synthetic number is a
                // valid `ConfigValue`: `serde_json::Number::deserialize` accepts it
                // (under `arbitrary_precision`) but rejects a genuine user object
                // with a different shape (see reviewer item 2 and the
                // `{"not":"a-number"}` rejection test). We then parse the number's
                // decimal token straight to `f32` rather than going through
                // `as_f64()`; the intermediate `f64` widening double-rounds values
                // near an `f32` midpoint and can pick the wrong neighbour (item 3).
                let number = serde_json::Number::deserialize(MapAccessDeserializer::new(map))?;
                let Ok(parsed) = number.to_string().parse::<f32>() else {
                    return Err(de::Error::custom(ValidationError::non_finite(
                        "value",
                        f32::NAN,
                    )));
                };
                de_finite_f32(parsed.into_deserializer()).map(ConfigValue::Float)
            }
        }

        deserializer.deserialize_any(ConfigValueVisitor)
    }
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
