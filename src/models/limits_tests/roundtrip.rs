// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn current_wire_fixtures_still_round_trip() {
    let spikes = IpcMessage::Spikes(SpikeBatch {
        session_id: Some("sess-1".into()),
        batch_id: 7,
        timestamp: 1_700_000_000,
        spikes: vec![SpikeEvent {
            channel: 3,
            time: 11,
            strength: 0.5,
        }],
        metadata: None,
    });
    let json = serde_json::to_value(&spikes).unwrap();
    let decoded: IpcMessage = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, spikes);
}
