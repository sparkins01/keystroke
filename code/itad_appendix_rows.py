"""
Produce the ITAD rows needed by the appendix tables, so M5 appears wherever
the other mechanisms do.

Covers:
  tab:appclip        full EER by gap (M5 has one condition, not two)
  tab:appksweep      EER by enrolment size and gap
  tab:particbalanced pooled against participant-balanced estimator
  tab:trialcounts    how M5's trial counts differ from M1 to M4

The participant-balanced estimator follows participant_balanced_eer.py: an
EER computed per test participant and then averaged, rather than pooled over
all trials, which checks whether participants contributing more trials are
driving the pooled figure.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_eer

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
ITAD_TEST = os.path.join(RESULT_DIR, 'with_itad', 'itad_trials_test.parquet')
MAIN_SUMMARY = os.path.join(RESULT_DIR, 'eer_by_mechanism_delta.csv')


def pooled_by(df, keys):
    rows = []
    for key, g in df.groupby(keys):
        eer, _ = compute_eer(g['label'], g['score'])
        rows.append(dict(zip(keys if isinstance(keys, list) else [keys],
                             key if isinstance(key, tuple) else (key,)),
                         eer=eer, n=len(g)))
    return pd.DataFrame(rows)


def participant_balanced(df, keys):
    """EER computed per participant then averaged over participants."""
    rows = []
    for key, g in df.groupby(keys):
        per = []
        for _pid, gp in g.groupby('participant_id'):
            if gp['label'].nunique() < 2:
                continue
            e, _ = compute_eer(gp['label'], gp['score'])
            if not np.isnan(e):
                per.append(e)
        rows.append(dict(zip(keys if isinstance(keys, list) else [keys],
                             key if isinstance(key, tuple) else (key,)),
                         eer_balanced=float(np.mean(per)) if per else np.nan,
                         n_participants=len(per)))
    return pd.DataFrame(rows)


def latex_row(label, values, fmt='%.1f'):
    cells = ' & '.join('--' if (v is None or (isinstance(v, float) and np.isnan(v)))
                       else fmt % v for v in values)
    return '%s & %s \\\\' % (label, cells)


def main():
    df = pd.read_parquet(ITAD_TEST,
                         columns=['k_enrollment', 'delta', 'label', 'score',
                                  'participant_id'])
    deltas = list(range(8))

    print('=' * 78)
    print('tab:appclip  -- M5 row (single condition; clipping cannot apply)')
    print('=' * 78)
    k5 = df[df['k_enrollment'] == 5]
    p = pooled_by(k5, ['delta']).set_index('delta')['eer'] * 100
    print(latex_row('M5 (ITAD) & Not applicable',
                    [p.get(d, np.nan) for d in deltas]))
    print()

    print('=' * 78)
    print('tab:appksweep  -- M5 rows')
    print('=' * 78)
    ks = pooled_by(df, ['k_enrollment', 'delta'])
    for k in [1, 3, 5]:
        sub = ks[ks['k_enrollment'] == k].set_index('delta')['eer'] * 100
        vals = [sub.get(d, np.nan) for d in deltas]
        label = ('\\multirow{3}{*}{M5 (ITAD)} & %d' % k) if k == 1 else ('                                       & %d' % k)
        print(latex_row(label, vals))
    print('(k=1 is undefined: a one-session gallery cannot form a profile)')
    print()

    print('=' * 78)
    print('tab:particbalanced  -- M5 rows')
    print('=' * 78)
    pooled = pooled_by(k5, ['delta']).set_index('delta')['eer'] * 100
    bal = participant_balanced(k5, ['delta']).set_index('delta')['eer_balanced'] * 100
    print(latex_row('\\multirow{2}{*}{M5 (ITAD)} & Pooled',
                    [pooled.get(d, np.nan) for d in deltas]))
    print(latex_row('                                       & Balanced',
                    [bal.get(d, np.nan) for d in deltas]))
    diff = (bal - pooled).abs().max()
    print('max absolute shift: %.2f points' % diff)
    print()

    print('=' * 78)
    print('tab:trialcounts  -- how M5 differs')
    print('=' * 78)
    counts = k5.groupby('delta').agg(
        genuine=('label', lambda s: int((s == 1).sum())),
        impostor=('label', lambda s: int((s == 0).sum())))
    counts['total'] = counts['genuine'] + counts['impostor']
    main_sum = pd.read_csv(MAIN_SUMMARY)
    main_sum = main_sum[(main_sum['mechanism'] == 'M1_Manhattan')
                        & (main_sum['clipped'] == 'clipped')
                        & (main_sum['k_enrollment'] == 5)].set_index('delta')
    counts['M1_total'] = main_sum['n_trials']
    counts['pct_of_M1'] = 100 * counts['total'] / counts['M1_total']
    print(counts.to_string(float_format=lambda v: '%.1f' % v))


if __name__ == '__main__':
    main()
