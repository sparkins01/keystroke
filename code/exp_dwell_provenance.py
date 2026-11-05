"""
Is the dwell signal behaviour or equipment?

exp_feature_ablation.py found that dwell alone verifies better than all four
digraph timings combined (23.0% against 24.8% EER at Delta=0) and ages far
more slowly (+8.4 points over seven weeks against +13.5). That is hard to
square with dwell being typing behaviour in this collection, because the
recorded values are mostly quantisation: about 26% of key events carry a
release timestamp identical to their press timestamp and the median hold is
about 4 ms, where real key holds run 70 to 150 ms.

The alternative is that the pattern of zero and near-zero dwell values is a
property of a participant's keyboard, polling rate or operating system. Each
participant kept the same equipment for the study, so such a pattern would
act as a device fingerprint: highly discriminative, and nearly static.

The study records a hardware change at week 5, which makes the two accounts
separable. A behavioural signal should not care about the hardware change
beyond the usual drift. A device signature should break at exactly that
point.

Three measurements:
  1. verification error by query week, dwell alone against digraphs alone,
     looking for a step at week 5 rather than a smooth trend;
  2. the share of each participant's dwell variance that sits between weeks
     rather than within them, which says how static the signal is;
  3. the per-participant rate of exactly-zero dwell, and how stable it is
     across weeks, which is the quantity a capture artefact would pin down.

Writes results/revision/dwell_provenance_*.csv and
results/figures/dwell_provenance.png.
"""
import os
import sys

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

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULT_DIR, 'revision')
FIG_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULT_DIR, 'figures'))

SUBSETS = {'dwell_only': ['M'], 'digraphs_only': ['DD', 'UD', 'DU', 'UU']}


def weights_for(subset):
    w = np.array([1.0 if f in subset else 0.0 for f in FEATURES])
    return w / w.sum()


def error_by_query_week(dataset, password_meta, folds, k_enrollment=5):
    rows = []
    for fold, train_ids, val_ids, test_ids in fold_splits(dataset.participant_ids, seed=0):
        if fold not in folds:
            continue
        matcher = ITADMatcher(fusion='equal', seed=0, min_occurrences=2)
        for name, subset in SUBSETS.items():
            matcher.weights = weights_for(subset)
            trials = generate_trials(
                dataset, matcher, None, test_ids, password_meta,
                name, fold, 'unclipped', k_enrollment=k_enrollment,
                max_impostors_per_trial=None, seed=0)
            df = pd.DataFrame(trials)
            df = df[np.isfinite(df['score'])]
            for qw, g in df.groupby('query_week'):
                if g['label'].nunique() < 2:
                    continue
                eer, _ = compute_eer(g['label'], g['score'])
                rows.append({'subset': name, 'fold': fold,
                             'query_week': int(qw), 'eer': eer, 'n': len(g)})
        print('  fold %d scored' % fold, flush=True)
    return pd.DataFrame(rows)


def dwell_structure(samples):
    """Per attempt: mean dwell and the fraction of its key events with
    exactly zero dwell."""
    rows = []
    for s in samples:
        press = np.array([c.press.value for c in s.char_events], dtype=np.float64) / 1e9
        release = np.array([c.release.value for c in s.char_events], dtype=np.float64) / 1e9
        d = release - press
        rows.append({'participant_id': s.participant_id, 'week': s.week,
                     'mean_dwell': float(d.mean()),
                     'zero_rate': float((d <= 0).mean())})
    return pd.DataFrame(rows)


def variance_split(df, column):
    """Share of total variance that lies between participants rather than
    within them. Close to 1 means the quantity is a near-static per-person
    constant."""
    overall = df[column].mean()
    per_p = df.groupby('participant_id')[column].agg(['mean', 'count'])
    between = float((per_p['count'] * (per_p['mean'] - overall) ** 2).sum())
    merged = df.merge(per_p[['mean']].rename(columns={'mean': '_pm'}),
                      on='participant_id')
    within = float(((merged[column] - merged['_pm']) ** 2).sum())
    total = between + within
    return between / total if total > 0 else np.nan


def plot(by_week, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    agg = by_week.groupby(['subset', 'query_week'])['eer'].mean().unstack(0) * 100
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(agg.index, agg['dwell_only'], marker='o', color='#d62728',
            label='dwell only')
    ax.plot(agg.index, agg['digraphs_only'], marker='s', color='#1f77b4',
            linestyle='--', label='four digraph timings')
    ax.axvline(5, color='black', linestyle=':', linewidth=1.2)
    ax.annotate('hardware change', xy=(5, ax.get_ylim()[1]), fontsize=8,
                ha='center', va='top')
    ax.set_xlabel('Query week')
    ax.set_ylabel('EER (%)')
    ax.set_title('Does the dwell signal break at the hardware change?', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('wrote %s' % out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = load_samples(use_cache=True, verbose=False)
    dataset = Dataset(samples)
    password_meta = {r['password_id']: r for r in build_password_metadata()}

    print('1. verification error by query week')
    by_week = error_by_query_week(dataset, password_meta, [0, 1, 2, 3, 4])
    by_week.to_csv(os.path.join(OUT_DIR, 'dwell_provenance_by_week.csv'), index=False)

    print()
    print('=' * 70)
    print('EER BY QUERY WEEK (mean across folds, %)')
    print('=' * 70)
    pivot = by_week.groupby(['subset', 'query_week'])['eer'].mean().unstack(0) * 100
    pivot['dwell_minus_digraph'] = pivot['dwell_only'] - pivot['digraphs_only']
    print(pivot.to_string(float_format=lambda v: '%.2f' % v))
    print()
    print('week-on-week change in dwell-only EER:')
    d = pivot['dwell_only'].diff()
    print(d.to_string(float_format=lambda v: '%+.2f' % v))
    print()

    print('=' * 70)
    print('2 AND 3. HOW STATIC IS THE DWELL SIGNAL?')
    print('=' * 70)
    ds = dwell_structure(samples)
    print('share of variance that is between participants rather than within:')
    print('  mean dwell        %.3f' % variance_split(ds, 'mean_dwell'))
    print('  zero-dwell rate   %.3f' % variance_split(ds, 'zero_rate'))
    print()
    per_p = ds.groupby('participant_id')['zero_rate'].agg(['mean', 'std'])
    print('per-participant zero-dwell rate: %d participants' % len(per_p))
    print('  mean across participants %.3f, spread between them %.3f'
          % (per_p['mean'].mean(), per_p['mean'].std()))
    print('  typical spread within one participant across weeks %.3f'
          % per_p['std'].mean())
    print()
    print('participants whose attempts are always fully zero-dwell: %d'
          % int((per_p['mean'] > 0.99).sum()))
    print('participants with essentially no zero dwell: %d'
          % int((per_p['mean'] < 0.01).sum()))
    ds.to_csv(os.path.join(OUT_DIR, 'dwell_provenance_structure.csv'), index=False)

    by_week_zero = ds.groupby('week')['zero_rate'].mean()
    print()
    print('mean zero-dwell rate by week:')
    print(by_week_zero.to_string(float_format=lambda v: '%.4f' % v))

    plot(by_week, os.path.join(FIG_DIR, 'dwell_provenance.png'))


if __name__ == '__main__':
    main()
