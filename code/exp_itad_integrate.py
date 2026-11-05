"""
Score ITAD across the full experimental grid, so it sits alongside M1 to M4
in the same tables, figures and model rather than in a section of its own.

Produces, in results/with_itad/:
  itad_trials_test.parquet, itad_trials_val.parquet
      trial-level scores in the same schema as the main run, so the DET
      panel and the mixed-effects model can consume them unchanged
  eer_by_mechanism_delta.csv
      the published summary with ITAD rows merged in
  mixed_model_input.parquet
      the model input with ITAD trials merged in

Clipping. The feature clipper applies to the aggregated representation used
by M1 and M2. ITAD scores percentiles within the enrolment profile, so a
monotone transform of the inputs leaves it unchanged and there is no clipped
against unclipped distinction to draw. Its rows carry the clipping label
'n/a' rather than being duplicated under both conditions.

Profile minimum. ITAD builds a profile for a graph only where four or more
occurrences exist. A gallery enrolled inside one week holds at most three
sessions once one is reserved for the zero-gap trials, which leaves the
longest gaps almost entirely undefined. The minimum is therefore relaxed to
two, which is recorded in the paper.
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
from metrics import summarize_by_mechanism_delta, compute_eer
from password_metadata import build_password_metadata
from matcher_itad import ITADMatcher

RESULTS = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULTS, 'with_itad')
MECH = 'M5_ITAD'


def generate(dataset, password_meta, k_values, folds, fusion, min_occ):
    test_rows, val_rows = [], []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=0):
        if fold not in folds:
            continue
        t0 = time.time()
        matcher = ITADMatcher(fusion=fusion, seed=0, min_occurrences=min_occ)
        matcher.fit(dataset, train_ids, None)
        for k in k_values:
            test_rows.extend(generate_trials(
                dataset, matcher, None, test_ids, password_meta, MECH, fold,
                'not_applicable', k_enrollment=k, max_impostors_per_trial=None, seed=0))
            val_rows.extend(generate_trials(
                dataset, matcher, None, val_ids, password_meta, MECH, fold,
                'not_applicable', k_enrollment=k, max_impostors_per_trial=None, seed=0))
        print('  fold %d scored in %.1f min' % (fold, (time.time() - t0) / 60),
              flush=True)
    return pd.DataFrame(test_rows), pd.DataFrame(val_rows)


def drop_undefined(df, label):
    before = len(df)
    df = df[np.isfinite(df['score'])].copy()
    print('%s: kept %d of %d trials (%.1f%% undefined at this profile minimum)'
          % (label, len(df), before, 100 * (before - len(df)) / before))
    return df


def build_model_rows(test, val):
    """is_error at a threshold calibrated per (fold, delta) on the validation
    participants, matching the convention used for M1 to M4."""
    thresholds = {}
    for (fold, delta), g in val.groupby(['fold', 'delta']):
        if g['label'].nunique() < 2:
            continue
        _, thr = compute_eer(g['label'], g['score'])
        thresholds[(fold, delta)] = thr

    thr = test.apply(lambda r: thresholds.get((r['fold'], r['delta'])), axis=1)
    out = test[thr.notna()].copy()
    thr = thr[thr.notna()]
    predicted_genuine = out['score'].values >= thr.values
    actual_genuine = out['label'].values == 1
    out['is_error'] = (predicted_genuine != actual_genuine).astype(float)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    parser.add_argument('--k-values', type=int, nargs='*', default=[1, 3, 5])
    parser.add_argument('--fusion', default='mdi', choices=['mdi', 'equal'])
    parser.add_argument('--min-occurrences', type=int, default=2)
    parser.add_argument('--n-bootstrap', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    password_meta = {r['password_id']: r for r in build_password_metadata()}
    print('scoring ITAD (%s fusion, profile minimum %d) at k = %s'
          % (args.fusion, args.min_occurrences, args.k_values))

    test, val = generate(dataset, password_meta, args.k_values, args.folds,
                         args.fusion, args.min_occurrences)
    test = drop_undefined(test, 'test')
    val = drop_undefined(val, 'val')

    test.to_parquet(os.path.join(OUT_DIR, 'itad_trials_test.parquet'), index=False)
    val.to_parquet(os.path.join(OUT_DIR, 'itad_trials_val.parquet'), index=False)
    print('wrote ITAD trial files')

    summary = summarize_by_mechanism_delta(test, n_boot=args.n_bootstrap, seed=0)
    published = pd.read_csv(os.path.join(RESULTS, 'eer_by_mechanism_delta.csv'))
    combined = pd.concat([published, summary], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, 'eer_by_mechanism_delta.csv'), index=False)
    print('wrote merged eer_by_mechanism_delta.csv (%d rows, %d new)'
          % (len(combined), len(summary)))

    model_rows = build_model_rows(test, val)
    base = pd.read_parquet(os.path.join(RESULTS, 'mixed_model_input.parquet'))
    keep = [c for c in base.columns if c in model_rows.columns]
    merged = pd.concat([base, model_rows[keep]], ignore_index=True)
    merged.to_parquet(os.path.join(OUT_DIR, 'mixed_model_input.parquet'), index=False)
    print('wrote merged mixed_model_input.parquet (%d rows, %d new)'
          % (len(merged), len(model_rows)))

    print()
    print('ITAD EER by gap, k=5:')
    k5 = summary[summary['k_enrollment'] == 5][
        ['delta', 'n_trials', 'eer_pooled', 'eer_ci_lo', 'eer_ci_hi',
         'fnmr_at_fmr_0.01']]
    print(k5.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print()
    print('total %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
