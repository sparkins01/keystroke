"""
Benchmark ITAD under this study's protocol, for TBIOM reviewer 2, point 5.

Runs the ITAD matcher of matcher_itad.py through the unchanged trial design
of protocol.py, so its numbers sit alongside M1 to M4 on the same genuine and
impostor trials, at the same enrolment sizes and the same temporal gaps.

Both fusion variants are run: equal weights across the five feature types,
and the random-forest importance weights the original paper uses.

Writes results/revision/itad_*.csv.
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
from metrics import compute_eer, compute_fnmr_at_fmr
from password_metadata import build_password_metadata
from matcher_itad import ITADMatcher, FEATURES

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')


def summarise(trials, mechanism):
    """ITAD is undefined where the gallery holds fewer than the paper's
    minimum of four occurrences per graph, which happens for the smallest
    enrolment weeks. Those trials score NaN and are dropped here rather than
    imputed, and the count is carried through so the comparison against
    M1 to M4 can state exactly which trials it rests on."""
    df = pd.DataFrame(trials)
    rows = []
    for (fold, delta), g in df.groupby(['fold', 'delta']):
        n_all = len(g)
        g = g[np.isfinite(g['score'])]
        n_used = len(g)
        if n_used == 0 or g['label'].nunique() < 2:
            continue
        eer, _ = compute_eer(g['label'], g['score'])
        rows.append({
            'mechanism': mechanism, 'fold': int(fold), 'delta': int(delta),
            'n_trials': n_used, 'n_trials_dropped': n_all - n_used,
            'pct_dropped': 100.0 * (n_all - n_used) / n_all,
            'eer': eer,
            'fnmr_at_fmr_0.01': compute_fnmr_at_fmr(g['label'], g['score'], 0.01),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--k-enrollment', type=int, default=5)
    parser.add_argument('--fusions', nargs='*', default=['equal', 'mdi'])
    parser.add_argument('--min-occurrences', type=int, nargs='*', default=[4, 2],
                        help='4 is the published minimum; 2 is a documented '
                             'relaxation so a full curve exists at every gap')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    meta_rows = build_password_metadata()
    password_meta = {r['password_id']: r for r in meta_rows}
    print('%d samples, %d participants, %d passwords'
          % (len(samples), len(dataset.participant_ids), len(dataset.password_ids)))
    print('k_enrollment = %d, so each graph profile holds %d values'
          % (args.k_enrollment, args.k_enrollment))
    print()

    all_rows = []
    weight_rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=0):
        if fold not in args.folds:
            continue
        for fusion in args.fusions:
          for min_occ in args.min_occurrences:
            t_f = time.time()
            mech = 'M5_ITAD_%s_min%d' % (fusion, min_occ)
            matcher = ITADMatcher(fusion=fusion, seed=0, min_occurrences=min_occ)
            matcher.fit(dataset, train_ids, None)
            if fusion == 'mdi':
                weight_rows.append(dict(zip(FEATURES, matcher.weights),
                                        fold=fold, min_occurrences=min_occ))
                print('  fold %d min%d MDI weights: %s' % (
                    fold, min_occ, ', '.join('%s=%.3f' % (f, w)
                                             for f, w in zip(FEATURES, matcher.weights))))

            trials = generate_trials(
                dataset, matcher, None, test_ids, password_meta, mech, fold,
                'unclipped', k_enrollment=args.k_enrollment,
                max_impostors_per_trial=None, seed=0)
            all_rows.extend(summarise(trials, mech))
            print('  fold %d %s scored in %.1f min'
                  % (fold, mech, (time.time() - t_f) / 60), flush=True)

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(os.path.join(OUT_DIR, 'itad_per_fold.csv'), index=False)

    summary = (per_fold.groupby(['mechanism', 'delta'])
               .agg(eer_mean=('eer', 'mean'), eer_sd=('eer', 'std'),
                    fnmr_at_1pct=('fnmr_at_fmr_0.01', 'mean'),
                    n_trials=('n_trials', 'sum'),
                    pct_dropped=('pct_dropped', 'mean'))
               .reset_index())
    summary.to_csv(os.path.join(OUT_DIR, 'itad.csv'), index=False)

    if weight_rows:
        w = pd.DataFrame(weight_rows)
        w.to_csv(os.path.join(OUT_DIR, 'itad_mdi_weights.csv'), index=False)
        print()
        print('MDI feature weights, mean across folds:')
        print(w[FEATURES].mean().to_string(float_format=lambda v: '%.4f' % v))

    print()
    print('=' * 66)
    print('ITAD EER BY TEMPORAL GAP (mean across folds)')
    print('=' * 66)
    print(summary.pivot(index='delta', columns='mechanism', values='eer_mean')
          .mul(100).to_string(float_format=lambda v: '%.2f' % v))
    print()

    published = os.path.join(os.path.dirname(OUT_DIR), 'eer_by_mechanism_delta.csv')
    if os.path.exists(published):
        ref = pd.read_csv(published)
        ref = ref[(ref['clipped'] == 'clipped')
                  & (ref['k_enrollment'] == args.k_enrollment)]
        print('for comparison, the published mechanisms at the same k:')
        print(ref.pivot(index='delta', columns='mechanism', values='eer_pooled')
              .mul(100).to_string(float_format=lambda v: '%.2f' % v))

    print()
    print('total %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
