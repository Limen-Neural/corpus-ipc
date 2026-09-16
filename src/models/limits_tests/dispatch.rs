// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn ipc_message_validate_dispatches_to_payload() {
    IpcMessage::Ping.validate().unwrap();
    IpcMessage::Shutdown.validate().unwrap();
    IpcMessage::TrainingComplete.validate().unwrap();
    IpcMessage::Loss(0.5).validate().unwrap();
}

#[test]
fn ipc_message_stimuli_dispatches_mask_mismatch() {
    let mut batch = StimulusBatch {
        values: vec![0.0],
        valid_mask: Some(vec![true, false]),
        ..Default::default()
    };
    assert!(IpcMessage::Stimuli(batch.clone()).validate().is_err());
    batch.valid_mask = Some(vec![true]);
    IpcMessage::Stimuli(batch).validate().unwrap();
}

#[test]
fn neuromodulator_message_validates() {
    let snapshot = NeuromodulatorSnapshot {
        tick: 1,
        dopamine: 0.1,
        cortisol: 0.2,
        acetylcholine: 0.3,
        tempo: 1.0,
    };
    IpcMessage::Neuromodulators(snapshot).validate().unwrap();
}
