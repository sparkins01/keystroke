"""
Section 3.1 "Raw sequence representation", used by the embedding models (M3/M4):
for a password of n characters, the ordered sequence of 2n key-event
timestamps (press and release), each represented as an inter-event duration
plus a press/release indicator.

Press and release events are chronologically interleaved (not simply
press[0],release[0],press[1],release[1],...) since a fast typist's key-down
for character i+1 can occur before the key-up for character i.
"""
import numpy as np


def compute_raw_sequence(char_events):
    """Returns an array of shape (2n, 2): column 0 = inter-event duration in
    seconds (0.0 for the first event), column 1 = press/release indicator
    (0.0 = press, 1.0 = release)."""
    events = []
    for e in char_events:
        events.append((e.press.value / 1e9, 0.0))
        events.append((e.release.value / 1e9, 1.0))
    events.sort(key=lambda x: x[0])

    timestamps = np.array([t for t, _ in events], dtype=np.float64)
    indicators = np.array([ind for _, ind in events], dtype=np.float64)
    durations = np.diff(timestamps, prepend=timestamps[0])

    return np.stack([durations, indicators], axis=1)
