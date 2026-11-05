"""
Separate password-set difficulty from password memorisation, using both
splits.

A single split cannot do this. Writing d for the intrinsic difficulty
difference between the two password halves, as that mechanism ranks them,
and m for the extra cost a mechanism pays on a password it never trained on:

    primary split  (train on A, unseen = B):   gap =  d + m
    swapped split  (train on B, unseen = A):   gap = -d + m

so m = (gap_primary + gap_swapped) / 2, and d cancels. Taking the difference
against the M1 control instead, as a single split invites, does not work,
because d is mechanism-specific: M1 and M3 do not rank password difficulty
the same way, so M1's d is not M3's d.

M1 is still the check on the design rather than the estimator. It fits
nothing per password, so its gap must flip sign exactly between the splits
and its m must come out at zero.

Writes results/revision/memorisation_estimate.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sstats

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')


def per_fold_gaps(tag):
    suffix = ('_' + tag) if tag else ''
    path = os.path.join(OUT_DIR, 'password_disjoint_per_fold%s.csv' % suffix)
    df = pd.read_csv(path)
    cell = (df.groupby(['mechanism', 'fold', 'password_set'])['eer']
            .mean().unstack('password_set'))
    cell['gap'] = cell['unseen'] - cell['seen']
    return cell.reset_index()[['mechanism', 'fold', 'gap']]


def main():
    a = per_fold_gaps('').rename(columns={'gap': 'gap_primary'})
    b = per_fold_gaps('swapped').rename(columns={'gap': 'gap_swapped'})
    both = a.merge(b, on=['mechanism', 'fold'])

    both['memorisation'] = (both['gap_primary'] + both['gap_swapped']) / 2.0
    both['difficulty'] = (both['gap_primary'] - both['gap_swapped']) / 2.0

    print('=' * 78)
    print('PER FOLD, IN EER POINTS')
    print('=' * 78)
    show = both.copy()
    for c in ['gap_primary', 'gap_swapped', 'memorisation', 'difficulty']:
        show[c] = show[c] * 100
    print(show.to_string(index=False, float_format=lambda v: '%+.3f' % v))
    print()

    print('=' * 78)
    print('MEMORISATION PENALTY, m = (primary + swapped) / 2')
    print('=' * 78)
    rows = []
    for mech, g in both.groupby('mechanism'):
        v = g['memorisation'].to_numpy() * 100
        t, p = sstats.ttest_1samp(v, 0.0)
        rows.append({
            'mechanism': mech,
            'memorisation_points': v.mean(),
            'sd': v.std(ddof=1),
            't': t, 'p': p,
            'positive_in_folds': '%d/%d' % (int((v > 0).sum()), len(v)),
            'difficulty_points': g['difficulty'].mean() * 100,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: '%.4g' % v))
    print()
    print('M1 is the control: it trains nothing per password, so its')
    print('memorisation term must be zero if the design is sound.')
    print()

    base = (pd.read_csv(os.path.join(OUT_DIR, 'password_disjoint_per_fold.csv'))
            .groupby('mechanism')['eer'].mean() * 100)
    print('penalty relative to each mechanism\'s own overall error:')
    for _, r in out.iterrows():
        mech = r['mechanism']
        print('  %-24s %+.3f points on %.1f%% EER  (%.1f%% relative)'
              % (mech, r['memorisation_points'], base[mech],
                 100 * r['memorisation_points'] / base[mech]))

    out.to_csv(os.path.join(OUT_DIR, 'memorisation_estimate.csv'), index=False)
    both.to_csv(os.path.join(OUT_DIR, 'memorisation_per_fold.csv'), index=False)
    print()
    print('wrote memorisation_estimate.csv')


if __name__ == '__main__':
    main()
