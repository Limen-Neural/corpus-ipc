// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn spike_events_accept_exact_max() {
    let limits = limits();
    let spikes = SpikeBatch {
        spikes: vec![
            SpikeEvent {
                channel: 0,
                time: 0,
                strength: 1.0
            };
            limits.max_spike_events
        ],
        ..Default::default()
    };
    spikes.validate_with(limits).expect("spike max");
}

#[test]
fn spike_events_reject_max_plus_one() {
    let limits = limits();
    let mut spikes = SpikeBatch {
        spikes: vec![
            SpikeEvent {
                channel: 0,
                time: 0,
                strength: 1.0
            };
            limits.max_spike_events
        ],
        ..Default::default()
    };
    spikes.spikes.push(SpikeEvent {
        channel: 1,
        time: 1,
        strength: 1.0,
    });
    let err = spikes.validate_with(limits).unwrap_err();
    assert_limit_exceeded(&err, "spikes");
}
