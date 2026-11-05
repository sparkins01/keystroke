"""
Does the length-by-Delta interaction hold within every mechanism?

exp_model_extensions.py fits a single length-by-Delta coefficient pooled over
all four mechanisms, and finds it positive and significant: longer passwords
age faster. The model-free slopes in exp_model_robustness.py do not agree
across mechanisms. M2 appears to run the other way, and M4 looks flat. A
single pooled coefficient would hide that.

This script settles it two ways:
  1. a likelihood-ratio test of the three-way Delta by length by mechanism
     term against the two-way model, which asks directly whether the
     mechanisms disagree;
  2. a separate fit per mechanism, which gives each one its own slope with a
     confidence interval rather than an average.

A finding that holds pooled but not within mechanism is a composition effect,
not a property of password length, and should be reported as such.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL, _fit_with_fallback
from exp_model_extensions import VC, lr_test, terms_of_interest

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
PRIMARY = os.path.join(RESULT_DIR, 'mixed_model_input.parquet')
OUT_DIR = os.path.join(RESULT_DIR, 'revision')

TWO_WAY = ("is_error ~ delta_c * mechanism + length_c + delta_c:length_c "
           "+ C(substitution_category) + hardware_change + C(fold)")
THREE_WAY = ("is_error ~ delta_c * mechanism * length_c "
             "+ C(substitution_category) + hardware_change + C(fold)")
PER_MECH = ("is_error ~ delta_c * length_c + C(substitution_category) "
            "+ hardware_change + C(fold)")


def prep(df, seed=0, max_rows=MAX_ROWS_FOR_MODEL):
    df = subsample(df, max_rows=max_rows, seed=seed).copy()
    for col in ['participant_id', 'password_id', 'fold', 'mechanism',
                'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error']:
        df[col] = df[col].astype(float)
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    df['dummy_group'] = 1
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_parquet(PRIMARY)
    length_mean_all = float(raw['length'].mean())

    print('=' * 76)
    print('DO THE MECHANISMS DISAGREE? THREE-WAY INTERACTION TEST')
    print('=' * 76)
    df = prep(raw)
    print('rows used: %d' % len(df))
    two, m1 = _fit_with_fallback(TWO_WAY, df, VC)
    three, m2 = _fit_with_fallback(THREE_WAY, df, VC)
    print('  two-way    llf=%14.2f  k=%d  (%s)' % (two.llf, len(two.params), m1))
    print('  three-way  llf=%14.2f  k=%d  (%s)' % (three.llf, len(three.params), m2))
    test = lr_test(three, two)
    print()
    print('  three-way vs two-way: LR=%.2f  df=%d  p=%.4g'
          % (test['lr_stat'], test['df'], test['p_value']))
    print()
    three_way_terms = [t for t in three.params.index if 'delta_c' in t and 'length_c' in t]
    print(terms_of_interest(three, three_way_terms).to_string(
        index=False, float_format=lambda v: '%.6g' % v))
    print()

    print('=' * 76)
    print('SEPARATE FIT PER MECHANISM')
    print('=' * 76)
    rows = []
    for mech in sorted(raw['mechanism'].unique()):
        sub = prep(raw[raw['mechanism'] == mech], max_rows=MAX_ROWS_FOR_MODEL // 2)
        result, method = _fit_with_fallback(PER_MECH, sub, VC)
        lm = float(raw[raw['mechanism'] == mech]['length'].mean())
        b_d = float(result.params['delta_c'])
        b_i = float(result.params['delta_c:length_c'])
        rows.append({
            'mechanism': mech,
            'n_rows': len(sub),
            'delta_c': b_d,
            'length_x_delta': b_i,
            'se': float(result.bse['delta_c:length_c']),
            'z': float(result.tvalues['delta_c:length_c']),
            'p': float(result.pvalues['delta_c:length_c']),
            'slope_6char': b_d + b_i * (6 - lm),
            'slope_12char': b_d + b_i * (12 - lm),
        })
        print('  %-14s fitted (%s)' % (mech, method))
    per_mech = pd.DataFrame(rows)
    print()
    print(per_mech.to_string(index=False, float_format=lambda v: '%.6g' % v))
    print()
    n_positive = int((per_mech['length_x_delta'] > 0).sum())
    n_sig = int(((per_mech['p'] < 0.05) & (per_mech['length_x_delta'] > 0)).sum())
    print('positive in %d of 4 mechanisms; positive and significant in %d of 4'
          % (n_positive, n_sig))

    per_mech.to_csv(os.path.join(OUT_DIR, 'length_by_mechanism.csv'), index=False)
    pd.DataFrame([test]).to_csv(
        os.path.join(OUT_DIR, 'length_three_way_test.csv'), index=False)
    print()
    print('wrote %s' % os.path.join(os.path.abspath(OUT_DIR), 'length_by_mechanism.csv'))


if __name__ == '__main__':
    main()
