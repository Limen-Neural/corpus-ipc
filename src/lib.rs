//! # spikenaut-backend
//!
//! `TraderBackend` trait and implementations for SNN signal processing.
//!
//! Provides a unified interface for routing normalised market signals through
//! different SNN backend implementations:
//!
//! - [`RustBackend`] — pure-Rust native backend (no external deps, always available)
//! - [`ZmqBrainBackend`] — Julia IPC backend via ZMQ SUB socket (feature `zmq`)
//!
//! Also includes [`NeroManifoldSnapshot`] — the 4-neuromodulator snapshot
//! packet parsed from the Julia brain's 88-byte NERO IPC frame.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
//! repository (closed-source). The backend abstraction layer connected a Rust SNN
//! to a Julia CUDA brain via ZMQ IPC in production before being open-sourced as a
//! standalone crate.
//!
//! ## References
//!
//! **NERO scoring (Neuromorphic Evaluation of Relevance and Orchestration):**
//!
//! The 4-element neuromodulator snapshot (`dopamine`, `cortisol`,
//! `acetylcholine`, `tempo`) follows the biologically motivated neuromodulator
//! framework:
//!
//! - Schultz, W. (1998). Predictive reward signal of dopamine neurons.
//!   *Journal of Neurophysiology*, 80(1), 1–27.
//!   *(Dopamine as prediction-error / reward signal)*
//! - Arnsten, A. F. T. (2009). Stress signalling pathways that impair
//!   prefrontal cortex structure and function. *Nature Reviews Neuroscience*,
//!   10(6), 410–422. *(Cortisol / stress-induced inhibition)*
//! - Hasselmo, M. E. (1999). Neuromodulation: acetylcholine and memory
//!   consolidation. *Trends in Cognitive Sciences*, 3(9), 351–359.
//!   *(Acetylcholine as attention / signal-to-noise modulator)*
//!
//! **ZMQ IPC wire format (88-byte NERO packet):**
//! - `[0..8]` — tick counter (`i64`, little-endian)
//! - `[8..72]` — 16 × `f32` lobe readout (little-endian)
//! - `[72..88]` — 4 × `f32` NERO scores: dopamine, cortisol, acetylcholine, tempo

pub mod telemetry;
pub mod error;
pub mod trait_def;
pub mod rust_backend;
pub mod models;

#[cfg(feature = "zmq")]
pub mod zmq_backend;

pub use error::BackendError;
pub use trait_def::{TraderBackend, BackendType};
pub use rust_backend::RustBackend;
pub use models::NeroManifoldSnapshot;
pub use telemetry::GpuTelemetry;

#[cfg(feature = "zmq")]
pub use zmq_backend::ZmqBrainBackend;
