"""
Revision experiment for TBIOM reviewer 1, points 1 and 3.

Reviewer 1 asks why M3, the least accurate mechanism, should be the one whose
ageing is slowest, and wants the trade-off between accuracy and ageing
examined more closely.

Section 5.1 of the paper offers a floor-effect account and marks it
unverified: if much of M3's error is already present at Delta=0 because the
representation is weak, there is less headroom left for drift to consume, so
its slope looks shallow for a reason that has nothing to do with recurrence.
This script tests that account.

The prediction is specific. Under a floor effect the ageing slope is a
function of baseline error, not of architecture. So:
  A. within a mechanism, cells that start with low error should age faster;
  B. across mechanisms, once baseline error is controlled for, the mechanism
     should add nothing.

The obvious way to test this is wrong. Splitting cells by their observed
Delta=0 error and then measuring how they change is a regression-to-the-mean
machine: cells that happened to look easy will drift worse and cells that
happened to look hard will drift better, whatever the truth. Baseline
difficulty is therefore measured on the validation participants, and the
slope on the test participants, so the two come from disjoint people.

Writes results/revision/floor_effect_*.csv and results/figures/floor_effect.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_eer

RESULTS = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
OUT_DIR = os.path.join(RESULTS, 'revision')
FIG_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULTS, 'figures'))

MECH_COLORS = {
    'M1_Manhattan': '#1f77b4', 'M2_ShallowML': '#2ca02c',
    'M3_TypeNet': '#9467bd', 'M4_TypeFormer': '#e377c2',
}
MECH_LABELS = {
    'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 Shallow ML',
    'M3_TypeNet': 'M3 TypeNet (LSTM)', 'M4_TypeFormer': 'M4 TypeFormer',
}


def baseline_difficulty():
    """EER at Delta=0 for each (fold, mechanism, password), measured on the
    validation participants, who appear nowhere in the slope estimate."""
    import pyarrow.parquet as pq
    val = pq.read_table(
        os.path.join(RESULTS, 'trials_val.parquet'),
        columns=['fold', 'mechanism', 'password_id', 'delta', 'label', 'score'],
        filters=[('k_enrollment', '=', 5), ('clipped', '=', 'clipped'),
                 ('delta', '=', 0)],
    ).to_pandas()
    rows = []
    for (fold, mech, pw), g in val.groupby(['fold', 'mechanism', 'password_id']):
        eer, _ = compute_eer(g['label'], g['score'])
        if np.isnan(eer):
            continue
        rows.append({'fold': fold, 'mechanism': mech, 'password_id': pw,
                     'baseline_eer': eer, 'n_val_trials': len(g)})
    return pd.DataFrame(rows)


def test_slopes():
    """Ageing slope for each (fold, mechanism, password) on the test
    participants, from the per-trial decision errors the paper models."""
    df = pd.read_parquet(
        os.path.join(RESULTS, 'mixed_model_input.parquet'),
        columns=['fold', 'mechanism', 'password_id', 'delta', 'is_error'])
    cell = (df.groupby(['fold', 'mechanism', 'password_id', 'delta'])['is_error']
            .mean().reset_index())
    rows = []
    for (fold, mech, pw), g in cell.groupby(['fold', 'mechanism', 'password_id']):
        g = g.sort_values('delta')
        if len(g) < 5:
            continue
        slope, intercept, r, p, se = sstats.linregress(g['delta'], g['is_error'])
        rows.append({'fold': fold, 'mechanism': mech, 'password_id': pw,
                     'slope': slope, 'slope_se': se,
                     'test_error_at_0': float(g.iloc[0]['is_error'])})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('measuring baseline difficulty on validation participants')
    base = baseline_difficulty()
    print('  cells: %d' % len(base))
    print('measuring ageing slope on test participants')
    slopes = test_slopes()
    print('  cells: %d' % len(slopes))

    cells = base.merge(slopes, on=['fold', 'mechanism', 'password_id'], how='inner')
    print('  matched cells: %d' % len(cells))
    cells.to_csv(os.path.join(OUT_DIR, 'floor_effect_cells.csv'), index=False)
    print()

    print('=' * 78)
    print('A. WITHIN EACH MECHANISM: DO EASY CELLS AGE FASTER?')
    print('=' * 78)
    print('Baseline from validation participants, slope from test participants,')
    print('so regression to the mean cannot manufacture the correlation.')
    print()
    rows = []
    for mech, g in cells.groupby('mechanism'):
        r, p = sstats.pearsonr(g['baseline_eer'], g['slope'])
        terciles = pd.qcut(g['baseline_eer'], 3, labels=['easy', 'middle', 'hard'])
        by_t = g.groupby(terciles, observed=True)['slope'].mean()
        rows.append({
            'mechanism': mech, 'n_cells': len(g),
            'pearson_r': r, 'p': p,
            'slope_easy': by_t.get('easy', np.nan),
            'slope_middle': by_t.get('middle', np.nan),
            'slope_hard': by_t.get('hard', np.nan),
        })
    within = pd.DataFrame(rows)
    print(within.to_string(index=False, float_format=lambda v: '%.5f' % v))
    print()
    print('A floor effect predicts a negative correlation and slope_easy > slope_hard.')
    print()

    print('=' * 78)
    print('B. ACROSS MECHANISMS: DOES MECHANISM SURVIVE CONTROLLING FOR BASELINE?')
    print('=' * 78)
    cells['baseline_c'] = cells['baseline_eer'] - cells['baseline_eer'].mean()
    m_base = smf.ols('slope ~ baseline_c + C(fold)', data=cells).fit()
    m_full = smf.ols('slope ~ baseline_c + C(mechanism) + C(fold)', data=cells).fit()
    lr = m_full.compare_f_test(m_base)
    print('  slope ~ baseline            R2=%.4f' % m_base.rsquared)
    print('  slope ~ baseline + mechanism R2=%.4f' % m_full.rsquared)
    print('  F test for mechanism: F=%.2f  df=%s  p=%.4g' % (lr[0], lr[2], lr[1]))
    print()
    terms = [t for t in m_full.params.index if 'mechanism' in t or t == 'baseline_c']
    tab = pd.DataFrame({
        'coef': m_full.params[terms], 'se': m_full.bse[terms],
        't': m_full.tvalues[terms], 'p': m_full.pvalues[terms],
    })
    print(tab.to_string(float_format=lambda v: '%.6g' % v))
    print()

    raw_means = cells.groupby('mechanism')['slope'].mean()
    adj = raw_means.copy()
    print('mechanism mean slope, raw versus baseline-adjusted:')
    ref = sorted(cells['mechanism'].unique())[0]
    for mech in sorted(cells['mechanism'].unique()):
        key = 'C(mechanism)[T.%s]' % mech
        delta_adj = float(m_full.params.get(key, 0.0))
        print('  %-14s raw %.5f   adjusted difference from %s: %+.5f'
              % (mech, raw_means[mech], ref, delta_adj))
    print()

    within.to_csv(os.path.join(OUT_DIR, 'floor_effect_within.csv'), index=False)
    tab.to_csv(os.path.join(OUT_DIR, 'floor_effect_across.csv'))

    plot(cells, os.path.join(FIG_DIR, 'floor_effect.png'))


def plot(cells, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mech, g in cells.groupby('mechanism'):
        ax.scatter(g['baseline_eer'] * 100, g['slope'] * 100, s=8, alpha=0.35,
                   color=MECH_COLORS[mech], label=MECH_LABELS[mech])
        if len(g) > 2:
            b = np.polyfit(g['baseline_eer'] * 100, g['slope'] * 100, 1)
            xs = np.linspace(g['baseline_eer'].min() * 100,
                             g['baseline_eer'].max() * 100, 20)
            ax.plot(xs, np.polyval(b, xs), color=MECH_COLORS[mech], linewidth=2)
    ax.set_xlabel('Baseline EER at $\\Delta$=0, validation participants (%)')
    ax.set_ylabel('Ageing slope, test participants (points per week)')
    ax.set_title('Does a high starting error flatten the slope?', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('wrote %s' % out_path)


if __name__ == '__main__':
    main()
