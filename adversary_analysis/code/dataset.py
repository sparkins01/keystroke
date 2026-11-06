"""
Precomputes, once for the whole population, everything downstream code needs
per sample: the aggregated feature vector (Section 3.1 bullet 2) and the raw
sequence representation (Section 3.1 bullet 1). Fold-specific work (IQR
clipping, model training) happens afterwards on top of this fixed cache.
"""
from collections import defaultdict

from features import compute_feature_vector
from raw_sequences import compute_raw_sequence


class Dataset:
    def __init__(self, samples):
        self.samples = samples
        self.feature_vectors = [compute_feature_vector(s.char_events) for s in samples]
        self.raw_sequences = [compute_raw_sequence(s.char_events) for s in samples]

        self.by_participant_password = defaultdict(list)
        self.by_password = defaultdict(list)
        for idx, s in enumerate(samples):
            self.by_participant_password[(s.participant_id, s.password_id)].append(idx)
            self.by_password[s.password_id].append(idx)

        self.participant_ids = sorted({s.participant_id for s in samples})
        self.password_ids = sorted({s.password_id for s in samples})

    def sessions_for(self, participant_id, password_id, max_week=None):
        """Indices of this participant's samples of this password, optionally
        restricted to week <= max_week, sorted by (week, repetition)."""
        idxs = self.by_participant_password.get((participant_id, password_id), [])
        if max_week is not None:
            idxs = [i for i in idxs if self.samples[i].week <= max_week]
        return sorted(idxs, key=lambda i: (self.samples[i].week, self.samples[i].repetition))

    def sessions_at_week(self, participant_id, password_id, week):
        idxs = self.by_participant_password.get((participant_id, password_id), [])
        return [i for i in idxs if self.samples[i].week == week]
