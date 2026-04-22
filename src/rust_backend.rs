//! Pure-Rust native backend — no external dependencies.

use crate::{BackendError, NeuralBackend};

/// Rust-native SNN backend.
///
/// Implements a simple push-pull encoding: each input channel is split
/// into a positive/negative pair. Channel `i` → `output[i*2]` (positive),
/// channel `i` → `output[i*2+1]` (negative magnitude).
///
/// Useful as a smoke-test stub and software fallback when Julia / ZMQ is
/// unavailable.
pub struct RustBackend {
    initialized: bool,
}

impl RustBackend {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for RustBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralBackend for RustBackend {
    fn process_signals(
        &mut self,
        inputs: &[f32],
    ) -> Result<Vec<f32>, BackendError> {
        if !self.initialized {
            return Err(BackendError::InitializationError(
                "RustBackend not initialized — call initialize() first".to_string(),
            ));
        }

        // Push-pull encoding: positive → bull channel; negative → bear channel.
        let mut output = vec![0.0f32; inputs.len() * 2];
        for i in 0..inputs.len() {
            let val = inputs[i];
            if val > 0.0 {
                output[i * 2]     = val;
                output[i * 2 + 1] = 0.0;
            } else {
                output[i * 2]     = 0.0;
                output[i * 2 + 1] = val.abs();
            }
        }
        Ok(output)
    }

    fn initialize(&mut self, _model_path: Option<&str>) -> Result<(), BackendError> {
        self.initialized = true;
        Ok(())
    }

    fn save_state(&self, _model_path: &str) -> Result<(), BackendError> {
        Ok(()) // no state to persist
    }

    fn get_spike_states(&self) -> Vec<bool> {
        // This backend is stateless, so it always returns an empty Vec.
        Vec::new()
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        // No internal state to reset.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_before_init_returns_error() {
        let mut b = RustBackend::new();
        assert!(b.process_signals(&[0.0; 4]).is_err());
    }

    #[test]
    fn positive_input_goes_to_bull_channel() {
        let mut b = RustBackend::new();
        b.initialize(None).unwrap();
        let inputs = vec![0.8, 0.0, 0.0];
        let out = b.process_signals(&inputs).unwrap();
        assert!((out[0] - 0.8).abs() < 1e-5);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn negative_input_goes_to_bear_channel() {
        let mut b = RustBackend::new();
        b.initialize(None).unwrap();
        let inputs = vec![0.0, 0.0, -0.5];
        let out = b.process_signals(&inputs).unwrap();
        assert_eq!(out[4], 0.0);          // bull channel for ch2
        assert!((out[5] - 0.5).abs() < 1e-5); // bear channel
    }

    #[test]
    fn output_has_double_the_elements() {
        let mut b = RustBackend::new();
        b.initialize(None).unwrap();
        let out = b.process_signals(&[0.1; 8]).unwrap();
        assert_eq!(out.len(), 16);
        let out_5 = b.process_signals(&[0.1; 5]).unwrap();
        assert_eq!(out_5.len(), 10);
    }

    #[test]
    fn spike_states_is_empty() {
        let b = RustBackend::new();
        assert!(b.get_spike_states().is_empty());
    }
}
