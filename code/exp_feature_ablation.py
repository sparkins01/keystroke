"""
How much of the signal is dwell, and is that defensible? For TBIOM reviewer 2,
point 4, and as a check on the ITAD benchmark of exp_itad.py.

Two observations prompted this.

First, in this collection dwell is close to degenerate. About 26% of key
events carry a release timestamp identical to their press timestamp, and the
median dwell is roughly 4 ms where real key holds run 70 to 150 ms. The
capture did not reliably record key-up.

Second, when ITAD's five feature types are fused with random-forest
importances, dwell takes between 46% and 57% of the weight, in every fold.

A feature that is mostly quantisation noise should not dominate. If it does,
the likeliest explanation is that the pattern of zero and near-zero dwell
values is a property of a participant's keyboard and polling rate rather
than of their typing, and in a longitudinal study where each participant
keeps the same equipment that behaves like a device fingerprint.

This script measures the size of the effect rather than asserting the
explanation. It scores the same trials three ways: all five feature types,
dwell alone, and the four digraph timings with dwell removed. If dwell alone
verifies well, and removing it costs a lot, the concern is real and the
paper should say so. If removing it costs little, the weight is an artefact
of the forest and nothing follows.

Writes results/revision/feature_ablation.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loading import load_samples
from dataset import Dataset
from splits import fold_splits
from protocol import generate_trials
from metrics import compute_eer
from password_metadata import build_password_metadata
from matcher_itad import ITADMatcher, FEATURES

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')

SUBSETS = {
    'all_five': FEATURES,
    'dwell_only': ['M'],
    'digraphs_only': ['DD', 'UD', 'DU', 'UU'],
}


def weights_for(subset):
    w = np.array([1.0 if f in subset else 0.0 for f in FEATURES])
    return w / w.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--k-enrollment', type=int, default=5)
    parser.add_argument('--min-occurrences', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    password_meta = {r['password_id']: r for r in build_password_metadata()}

    rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=0):
        if fold not in args.folds:
            continue
        # one matcher per fold, reused across subsets so the cached per-sample
        # feature arrays are computed once
        matcher = ITADMatcher(fusion='equal', seed=0,
                              min_occurrences=args.min_occurrences)
        for name, subset in SUBSETS.items():
            matcher.weights = weights_for(subset)
            trials = generate_trials(
                dataset, matcher, None, test_ids, password_meta,
                'ITAD_%s' % name, fold, 'unclipped',
                k_enrollment=args.k_enrollment,
                max_impostors_per_trial=None, seed=0)
            df = pd.DataFrame(trials)
            df = df[np.isfinite(df['score'])]
            for delta, g in df.groupby('delta'):
                if g['label'].nunique() < 2:
                    continue
                eer, _ = compute_eer(g['label'], g['score'])
                rows.append({'subset': name, 'fold': fold, 'delta': int(delta),
                             'eer': eer, 'n_trials': len(g)})
            print('  fold %d %-14s scored' % (fold, name), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'feature_ablation_per_fold.csv'), index=False)

    summary = (out.groupby(['subset', 'delta'])
               .agg(eer_mean=('eer', 'mean'), eer_sd=('eer', 'std'))
               .reset_index())
    summary.to_csv(os.path.join(OUT_DIR, 'feature_ablation.csv'), index=False)

    print()
    print('=' * 66)
    print('EER BY FEATURE SUBSET (mean across folds, %)')
    print('=' * 66)
    pivot = summary.pivot(index='delta', columns='subset', values='eer_mean') * 100
    pivot = pivot[['all_five', 'dwell_only', 'digraphs_only']]
    pivot['cost_of_removing_dwell'] = pivot['digraphs_only'] - pivot['all_five']
    print(pivot.to_string(float_format=lambda v: '%.2f' % v))
    print()
    print('A dwell_only column near 50% would mean dwell carries no identity')
    print('information. Much below 50% means it does, despite the degeneracy.')
    print()
    print('total %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
