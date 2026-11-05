"""
Is the seen-to-unseen gap real, and is it larger for the learnt models than
for a matcher that cannot memorise a password?

The raw gaps are all slightly negative, meaning unseen passwords come out a
little easier. That is not evidence about memorisation on its own, because
the two password halves need not be exactly equal in difficulty despite the
stratified split. M1 measures that residual difference: it fits only a
population variability estimate, so it has no way to memorise a password and
whatever gap it shows is the password sets, not learning.

The quantity of interest is therefore each learnt model's gap minus M1's gap,
on the same fold. If TypeNet or TypeFormer were leaning on password-specific
patterns, their gap would be more positive than M1's, meaning unseen
passwords cost them more than the password sets alone explain.

Folds are the unit of analysis and the comparison is paired within fold.
Five folds is few, so a Wilcoxon signed-rank test is reported next to the
t-test and the per-fold values are printed in full rather than summarised
away.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as sstats

OUT_DIR = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'revision')
CONTROL = 'M1_Manhattan_unclipped'


def load(tag):
    suffix = ('_' + tag) if tag else ''
    path = os.path.join(OUT_DIR, 'password_disjoint_per_fold%s.csv' % suffix)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def gaps_per_fold(df):
    """Mean EER over gaps, per (mechanism, fold, password set), then the
    unseen-minus-seen difference."""
    cell = (df.groupby(['mechanism', 'fold', 'password_set'])['eer']
            .mean().unstack('password_set'))
    cell['gap'] = cell['unseen'] - cell['seen']
    return cell.reset_index()


def report(df, label):
    print('=' * 78)
    print('%s' % label)
    print('=' * 78)
    g = gaps_per_fold(df)

    print('mean EER by mechanism and password set:')
    summary = g.groupby('mechanism')[['seen', 'unseen', 'gap']].mean()
    print((summary * 100).to_string(float_format=lambda v: '%+.3f' % v))
    print()

    wide = g.pivot(index='fold', columns='mechanism', values='gap') * 100
    print('unseen minus seen, per fold (EER points, negative = unseen easier):')
    print(wide.to_string(float_format=lambda v: '%+.3f' % v))
    print()

    rows = []
    for mech in wide.columns:
        v = wide[mech].to_numpy()
        t, p_t = sstats.ttest_1samp(v, 0.0)
        rows.append({'mechanism': mech, 'mean_gap': v.mean(),
                     'sd': v.std(ddof=1), 't': t, 'p_vs_zero': p_t})
    print('is the gap different from zero?')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: '%.4g' % v))
    print()

    print('relative to the %s control, paired within fold:' % CONTROL)
    print('(positive means unseen passwords cost this mechanism more than the')
    print(' password sets alone explain, which is what memorisation would do)')
    rows = []
    for mech in wide.columns:
        if mech == CONTROL:
            continue
        d = (wide[mech] - wide[CONTROL]).to_numpy()
        t, p_t = sstats.ttest_1samp(d, 0.0)
        try:
            _, p_w = sstats.wilcoxon(d)
        except ValueError:
            p_w = np.nan
        rows.append({
            'mechanism': mech, 'mean_excess_gap': d.mean(), 'sd': d.std(ddof=1),
            't': t, 'p_t': p_t, 'p_wilcoxon': p_w,
            'positive_in_folds': '%d/%d' % (int((d > 0).sum()), len(d)),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: '%.4g' % v))
    print()
    return g, out


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    collected = []
    for tag, label in [('', 'PRIMARY SPLIT'), ('swapped', 'SWAPPED SPLIT')]:
        df = load(tag)
        if df is None:
            print('(%s not available yet)' % (label.lower()))
            print()
            continue
        g, excess = report(df, label)
        g['split'] = tag or 'primary'
        excess['split'] = tag or 'primary'
        collected.append((g, excess))

    if len(collected) == 2:
        print('=' * 78)
        print('DO THE TWO SPLITS AGREE?')
        print('=' * 78)
        both = pd.concat([c[1] for c in collected], ignore_index=True)
        pivot = both.pivot(index='mechanism', columns='split',
                           values='mean_excess_gap')
        pivot['same_sign'] = (np.sign(pivot['primary'])
                              == np.sign(pivot['swapped']))
        print('excess gap over the control, EER points, by split:')
        print(pivot.to_string(float_format=lambda v: '%+.3f' % v))
        print()
        allg = pd.concat([c[0] for c in collected], ignore_index=True)
        allg.to_csv(os.path.join(OUT_DIR, 'password_disjoint_gaps.csv'), index=False)
        both.to_csv(os.path.join(OUT_DIR, 'password_disjoint_excess.csv'), index=False)
        print('wrote password_disjoint_gaps.csv and password_disjoint_excess.csv')
    elif collected:
        collected[0][0].to_csv(
            os.path.join(OUT_DIR, 'password_disjoint_gaps.csv'), index=False)
        collected[0][1].to_csv(
            os.path.join(OUT_DIR, 'password_disjoint_excess.csv'), index=False)


if __name__ == '__main__':
    main()
