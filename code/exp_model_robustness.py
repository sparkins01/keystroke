"""
Robustness checks for the two extended model terms (reviewer 2, points 2, 6
and 8).

Two things in exp_model_extensions.py need checking before they are written
up.

First, the enrolment-week by Delta interaction rests on badly unbalanced
support. The trial grid is triangular, so enrolment week 8 contributes only
Delta=0 and enrolment week 7 only Delta 0 and 1. A slope estimated for those
weeks rests on one or two points. This script refits on the balanced
rectangle of enrolment weeks 1 to 4 against Delta 0 to 4, where every cell is
populated, and asks whether the interaction survives.

Second, a coefficient on a mean-centred interaction is hard to read. This
script converts it into the quantity the reviewer actually asked for: the
ageing slope in error-probability points per week, separately for a
6-character and a 12-character password. It also plots the model-free error
curve per length, so the claim does not rest on the model alone.

Writes results/revision/model_robustness_*.csv and
results/figures/length_ageing.png.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL, _fit_with_fallback
from exp_model_extensions import SPECS, VC, prepare, lr_test, terms_of_interest

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
PRIMARY = os.path.join(RESULT_DIR, 'mixed_model_input.parquet')
OUT_DIR = os.path.join(RESULT_DIR, 'revision')
FIG_DIR = os.environ.get('KD_FIG_DIR', os.path.join(RESULT_DIR, 'figures'))

MECH_COLORS = {
    'M1_Manhattan': '#1f77b4', 'M2_ShallowML': '#2ca02c',
    'M3_TypeNet': '#9467bd', 'M4_TypeFormer': '#e377c2',
}
MECH_LABELS = {
    'M1_Manhattan': 'M1 Manhattan', 'M2_ShallowML': 'M2 Shallow ML',
    'M3_TypeNet': 'M3 TypeNet (LSTM)', 'M4_TypeFormer': 'M4 TypeFormer',
}


def balanced_rectangle_check(max_rows, seed):
    """Refit on enrolment weeks 1 to 4 against Delta 0 to 4, where the grid is
    complete, so the interaction cannot be an artefact of ragged support."""
    df = pd.read_parquet(PRIMARY)
    before = len(df)
    df = df[(df['enrol_week'] <= 4) & (df['delta'] <= 4)].copy()
    print('balanced rectangle: %d of %d rows (enrol week 1-4, delta 0-4)'
          % (len(df), before))

    df = subsample(df, max_rows=max_rows, seed=seed).copy()
    for col in ['participant_id', 'password_id', 'fold', 'mechanism',
                'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error', 'enrol_week']:
        df[col] = df[col].astype(float)
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    df['enrol_week_c'] = df['enrol_week'] - df['enrol_week'].mean()
    df['dummy_group'] = 1

    grid = df.groupby(['enrol_week', 'delta']).size().unstack(fill_value=0)
    print('cells populated: %d of %d' % (int((grid > 0).sum().sum()), grid.size))
    print('correlation(enrol_week, delta) = %+.3f'
          % np.corrcoef(df['enrol_week'], df['delta'])[0, 1])
    print('rows used: %d' % len(df))
    print()

    fits = {}
    for name in ['S0_published', 'S3_both']:
        result, method = _fit_with_fallback(SPECS[name], df, VC)
        fits[name] = result
        print('  %-20s llf=%14.2f  (%s)' % (name, result.llf, method))

    comparison = lr_test(fits['S3_both'], fits['S0_published'])
    print()
    print('  S3 vs S0 on the balanced rectangle: LR=%.2f df=%d p=%.3g'
          % (comparison['lr_stat'], comparison['df'], comparison['p_value']))
    print()

    want = ['delta_c', 'length_c', 'delta_c:length_c', 'enrol_week_c',
            'enrol_week_c:delta_c']
    tab = terms_of_interest(fits['S3_both'], want)
    print(tab.to_string(index=False, float_format=lambda v: '%.6g' % v))
    tab['subset'] = 'balanced_rectangle'
    return tab, comparison, df


def length_slopes(terms_csv, length_mean):
    """Turn the centred interaction into a slope per password length, which
    is the form the reviewer's question is posed in."""
    terms = pd.read_csv(terms_csv)
    rows = []
    for link in ['linear', 'logit']:
        sub = terms[(terms['link'] == link) & (terms['spec'] == 'S1_length_x_delta')]
        if sub.empty:
            continue
        b_delta = sub[sub['term'] == 'delta_c']['coef'].iloc[0]
        se_delta = sub[sub['term'] == 'delta_c']['se'].iloc[0]
        b_int = sub[sub['term'] == 'delta_c:length_c']['coef'].iloc[0]
        se_int = sub[sub['term'] == 'delta_c:length_c']['se'].iloc[0]
        for length in [6, 8, 10, 12]:
            offset = length - length_mean
            slope = b_delta + b_int * offset
            # the two terms are correlated, so this standard error is
            # indicative rather than exact
            se = np.sqrt(se_delta ** 2 + (offset * se_int) ** 2)
            rows.append({
                'link': link, 'length': length, 'slope_per_week': slope,
                'se_indicative': se,
                'slope_over_7_weeks': slope * 7,
            })
    return pd.DataFrame(rows)


def model_free_length_curves():
    """Mean decision error by (length, Delta), with no model in the way."""
    df = pd.read_parquet(PRIMARY, columns=['mechanism', 'length', 'delta',
                                           'is_error', 'participant_id'])
    by_len = (df.groupby(['length', 'delta'])['is_error'].mean()
              .rename('error').reset_index())
    by_mech = (df.groupby(['mechanism', 'length', 'delta'])['is_error'].mean()
               .rename('error').reset_index())

    rows = []
    for (mech, length), g in by_mech.groupby(['mechanism', 'length']):
        g = g.sort_values('delta')
        slope, intercept, r, p, se = sstats.linregress(g['delta'], g['error'])
        rows.append({'mechanism': mech, 'length': int(length),
                     'slope_per_week': slope, 'p': p,
                     'error_at_delta0': float(g[g['delta'] == 0]['error'].iloc[0])})
    slopes = pd.DataFrame(rows)
    return by_len, by_mech, slopes


def plot_length(by_len, by_mech, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lengths = sorted(by_len['length'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    for i, length in enumerate(lengths):
        g = by_len[by_len['length'] == length].sort_values('delta')
        shade = 0.15 + 0.8 * i / max(1, len(lengths) - 1)
        ax.plot(g['delta'], g['error'] * 100, marker='o', markersize=4,
                color=plt.cm.viridis(shade), label='%d chars' % length)
    ax.set_xlabel(r'Temporal gap $\Delta$ (weeks)')
    ax.set_ylabel('Decision error (%)')
    ax.set_title('Ageing by password length, all mechanisms', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for mech in sorted(by_mech['mechanism'].unique()):
        sub = by_mech[by_mech['mechanism'] == mech]
        slopes = []
        for length in lengths:
            g = sub[sub['length'] == length].sort_values('delta')
            slope = sstats.linregress(g['delta'], g['error'])[0]
            slopes.append(slope * 100)
        ax.plot(lengths, slopes, marker='o', color=MECH_COLORS[mech],
                label=MECH_LABELS[mech])
    ax.set_xlabel('Password length (characters)')
    ax.set_ylabel('Ageing slope (error points per week)')
    ax.set_title('Longer passwords age faster', fontsize=10)
    ax.set_xticks(lengths)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('wrote %s' % out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 76)
    print('BALANCED-RECTANGLE REFIT OF THE ENROLMENT-WEEK TERMS')
    print('=' * 76)
    tab, comparison, df = balanced_rectangle_check(MAX_ROWS_FOR_MODEL, 0)
    tab.to_csv(os.path.join(OUT_DIR, 'model_robustness_balanced.csv'), index=False)
    print()

    full = pd.read_parquet(PRIMARY, columns=['length'])
    length_mean = float(full['length'].mean())
    print('=' * 76)
    print('AGEING SLOPE BY PASSWORD LENGTH (length mean = %.3f)' % length_mean)
    print('=' * 76)
    slopes = length_slopes(
        os.path.join(OUT_DIR, 'model_extensions_terms_primary.csv'), length_mean)
    print(slopes.to_string(index=False, float_format=lambda v: '%.6g' % v))
    slopes.to_csv(os.path.join(OUT_DIR, 'model_length_slopes.csv'), index=False)
    print()

    print('=' * 76)
    print('MODEL-FREE ERROR CURVES BY LENGTH')
    print('=' * 76)
    by_len, by_mech, mf_slopes = model_free_length_curves()
    pivot = by_len.pivot(index='delta', columns='length', values='error')
    print((pivot * 100).to_string(float_format=lambda v: '%.2f' % v))
    print()
    print('fitted slope per length, per mechanism (error points per week):')
    print((mf_slopes.assign(slope_per_week=mf_slopes['slope_per_week'] * 100)
           .pivot(index='length', columns='mechanism', values='slope_per_week')
           .to_string(float_format=lambda v: '%.3f' % v)))
    by_len.to_csv(os.path.join(OUT_DIR, 'length_curves_pooled.csv'), index=False)
    by_mech.to_csv(os.path.join(OUT_DIR, 'length_curves_by_mechanism.csv'), index=False)
    mf_slopes.to_csv(os.path.join(OUT_DIR, 'length_slopes_model_free.csv'), index=False)
    print()
    plot_length(by_len, by_mech, os.path.join(FIG_DIR, 'length_ageing.png'))


if __name__ == '__main__':
    main()
