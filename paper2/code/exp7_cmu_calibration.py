"""
Experiment 7: calibrate our matcher against a published benchmark.

The most likely objection to this paper is that our absolute error rates sit
well above published fixed-text figures, so the negative result might reflect a
weak implementation rather than the modality. This experiment answers it
directly by running our own M1 on Killourhy and Maxion's public benchmark under
their own protocol, where the correct answer is already known.

Killourhy, K.S. and Maxion, R.A. (2009) "Comparing Anomaly-Detection Algorithms
for Keystroke Dynamics", DSN 2009, report an equal error rate of 0.0962 for the
scaled Manhattan detector. Their protocol:

  - 51 subjects typing ".tie5Roanl", 8 sessions of 50 repetitions each
  - train on the first 200 repetitions of the genuine subject
  - genuine test scores: that subject's remaining 200 repetitions
  - impostor test scores: the first 5 repetitions from each of the other
    50 subjects
  - one equal error rate per subject, then averaged over the 51 subjects

Three configurations are run:

  1. Their scaled Manhattan exactly as published, which checks that the harness
     itself reproduces the reference number.
  2. Our M1, unmodified, under the same protocol. If this lands near 0.0962 the
     implementation is sound and the gap on our own corpus is a property of the
     protocol rather than of the code.
  3. Our M1 with only 5 enrolment samples rather than 200, which isolates how
     much of that gap is simply the cost of a realistic enrolment size.
  4. The same, but with the five samples taken immediately before the test set
     rather than at the very start. Configuration 3 changes the enrolment size
     and the enrolment-to-test gap together, so this separates them.

The equal error rate uses the same `compute_eer` as every other result in this
paper, so the comparison is like for like.
"""
import os
import sys

import numpy as np
import pandas as pd

from metrics import compute_eer

from matchers_classical import M1Manhattan

DATA = os.environ.get('KD_CMU_CSV', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_cmu', 'DSL-StrongPasswordData.csv'))
RESULT_DIR = os.environ.get('KD_P2_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
PUBLISHED_EER = 0.0962

N_TRAIN = 200        # first 200 repetitions, as in the reference protocol
N_IMPOSTOR = 5       # first 5 repetitions from each other subject


def load():
    df = pd.read_csv(DATA)
    feats = [c for c in df.columns if c not in ('subject', 'sessionIndex', 'rep')]
    df = df.sort_values(['subject', 'sessionIndex', 'rep']).reset_index(drop=True)
    return df, feats


# ---------------------------------------------------------------- detectors

def km_scaled_manhattan(train, query):
    """The published detector: absolute deviation from the training mean,
    divided by the training mean absolute deviation, summed."""
    mean = train.mean(axis=0)
    mad = np.abs(train - mean).mean(axis=0)
    mad = np.where(mad <= 0, 1e-9, mad)
    return -(np.abs(query - mean) / mad).sum(axis=1)


class _StubDataset:
    """The smallest object satisfying the interface M1Manhattan expects, so the
    matcher is exercised unmodified rather than reimplemented here."""

    def __init__(self, vectors, subjects, password_id='cmu'):
        self.feature_vectors = list(vectors)
        self.password_ids = [password_id]
        self.by_password = {password_id: list(range(len(vectors)))}

        class _S:
            def __init__(self, pid):
                self.participant_id = pid
                self.password_id = password_id

        self.samples = [_S(s) for s in subjects]


class _NoOpClipper:
    def transform(self, vector, password_id):
        return vector


def our_m1(train_vecs, query_vecs, all_train_vecs, all_train_subjects):
    """Our M1, unmodified. Its per-feature scale comes from within-gallery
    variability with a population-level floor, so the population statistics are
    built from every subject's training portion, which is the analogue of the
    training fold in our own protocol."""
    idx_offset = len(all_train_vecs)
    vectors = list(all_train_vecs) + list(train_vecs) + list(query_vecs)
    subjects = list(all_train_subjects) + ['__gallery__'] * len(train_vecs) \
        + ['__query__'] * len(query_vecs)
    ds = _StubDataset(vectors, subjects)
    clipper = _NoOpClipper()

    m1 = M1Manhattan()
    m1.fit(ds, set(all_train_subjects), clipper)

    gallery_idx = list(range(idx_offset, idx_offset + len(train_vecs)))
    query_idx = list(range(idx_offset + len(train_vecs), len(vectors)))
    gallery = m1.gallery(ds, clipper, gallery_idx)
    return np.array(m1.score_many(ds, clipper, gallery, query_idx))


# ---------------------------------------------------------------- protocol

def evaluate(df, feats, detector, n_enrol=N_TRAIN):
    """n_enrol is a count, or 'late5' for five samples adjacent to the test set."""
    subjects = sorted(df['subject'].unique())
    by_subject = {s: g[feats].to_numpy(dtype=float) for s, g in df.groupby('subject')}

    # population reference for our M1: every subject's training portion
    all_train = np.vstack([by_subject[s][:N_TRAIN] for s in subjects])
    all_train_subjects = [s for s in subjects for _ in range(N_TRAIN)]

    eers = []
    for s in subjects:
        X = by_subject[s]
        if n_enrol == 'late5':
            # five samples taken immediately before the test set, which holds
            # the temporal gap fixed and varies only the enrolment size
            train = X[N_TRAIN - 5:N_TRAIN]
        else:
            train = X[:n_enrol]
        genuine = X[N_TRAIN:]
        impostor = np.vstack([by_subject[o][:N_IMPOSTOR] for o in subjects if o != s])

        if detector == 'km':
            g = km_scaled_manhattan(train, genuine)
            i = km_scaled_manhattan(train, impostor)
        else:
            scores = our_m1(train, np.vstack([genuine, impostor]),
                            all_train, all_train_subjects)
            g, i = scores[:len(genuine)], scores[len(genuine):]

        labels = np.r_[np.ones(len(g)), np.zeros(len(i))]
        eer, _ = compute_eer(labels, np.r_[g, i])
        eers.append(eer)
    return np.array(eers)


def main():
    df, feats = load()
    subjects = sorted(df['subject'].unique())
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('=' * 76)
    emit('EXPERIMENT 7: calibration against the Killourhy & Maxion benchmark')
    emit('=' * 76)
    emit(f'{len(subjects)} subjects, {len(feats)} features, '
         f'{len(df) // len(subjects)} repetitions each')
    emit(f'protocol: {N_TRAIN} enrolment, remaining genuine, '
         f'{N_IMPOSTOR} repetitions from each of the other subjects')
    emit(f'reference: scaled Manhattan, EER = {PUBLISHED_EER:.4f}')
    emit()

    runs = [
        ('Killourhy & Maxion scaled Manhattan', 'km', N_TRAIN),
        ('Our M1, same protocol', 'm1', N_TRAIN),
        ('Our M1, first 5 enrolment samples', 'm1', 5),
        ('Our M1, last 5 before test set', 'm1', 'late5'),
    ]
    rows = []
    for label, det, n in runs:
        eers = evaluate(df, feats, det, n_enrol=n)
        rows.append({'configuration': label, 'n_enrolment': str(n),
                     'eer_mean': eers.mean(), 'eer_sd': eers.std(ddof=1),
                     'eer_median': np.median(eers), 'n_subjects': len(eers)})
        emit(f'{label:36} EER {eers.mean():.4f}  '
             f'(sd {eers.std(ddof=1):.4f}, median {np.median(eers):.4f})')
    emit()

    tab = pd.DataFrame(rows)
    km = tab.iloc[0]['eer_mean']
    ours = tab.iloc[1]['eer_mean']
    small = tab.iloc[2]['eer_mean']
    late = tab.iloc[3]['eer_mean']

    emit('-' * 76)
    emit('Reading')
    emit('-' * 76)
    emit(f'harness reproduces the published figure to within '
         f'{abs(km - PUBLISHED_EER):.4f} ({abs(km - PUBLISHED_EER) / PUBLISHED_EER * 100:.1f}%)')
    emit(f'our M1 differs from the published detector by {abs(ours - km):.4f}')
    emit(f'reducing enrolment from {N_TRAIN} to 5 samples costs '
         f'{(small - ours) * 100:+.2f} percentage points')
    emit(f'  of which enrolment size alone accounts for {(late - ours) * 100:+.2f} points')
    emit(f'  and the added temporal gap for {(small - late) * 100:+.2f} points')
    verdict = ('our implementation reproduces the benchmark, so the higher error '
               'rates on our own corpus are a property of the protocol and the '
               'enrolment size rather than of the code'
               if abs(ours - PUBLISHED_EER) < 0.03 else
               'our implementation does NOT reproduce the benchmark; investigate '
               'before relying on the absolute figures')
    emit(f'=> {verdict}')

    os.makedirs(RESULT_DIR, exist_ok=True)
    tab.to_csv(os.path.join(RESULT_DIR, 'exp7_cmu_calibration.csv'), index=False)
    with open(os.path.join(RESULT_DIR, 'exp7_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    emit()
    emit(f'written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
