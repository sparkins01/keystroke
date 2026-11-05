"""
Refit the mixed-effects model over all five mechanisms, and report the
additional covariates alongside the published ones.

Reads the merged model input from results/with_itad, so ITAD enters the model
on the same footing as M1 to M4 rather than being discussed separately.

Three specifications:
  main        the published main-effects model, now with five mechanisms
  interaction main plus delta by mechanism, for the likelihood-ratio test
  extended    interaction plus enrolment week and length by delta

Prints the coefficients needed for the results tables.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL, _fit_with_fallback

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
MERGED = os.path.join(RESULT_DIR, 'with_itad', 'mixed_model_input.parquet')
OUT_DIR = os.path.join(RESULT_DIR, 'revision')

VC = {'participant': '0 + C(participant_id)', 'password': '0 + C(password_id)'}

BASE_TERMS = ("length_c + C(substitution_category) + hardware_change + C(fold)")
SPECS = {
    'main': "is_error ~ delta_c + mechanism + " + BASE_TERMS,
    'interaction': "is_error ~ delta_c * mechanism + " + BASE_TERMS,
    'extended': ("is_error ~ delta_c * mechanism + enrol_week_c "
                 "+ delta_c:length_c + " + BASE_TERMS),
}


def prepare(path, k, clipped_values, max_rows, seed):
    df = pd.read_parquet(path)
    df = df[(df['k_enrollment'] == k) & (df['clipped'].isin(clipped_values))]
    print('rows available: %d   mechanisms: %s'
          % (len(df), sorted(df['mechanism'].unique())))
    df = subsample(df, max_rows=max_rows, seed=seed).copy()
    for col in ['participant_id', 'password_id', 'fold', 'mechanism',
                'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error', 'enrol_week']:
        df[col] = df[col].astype(float)
    for col, src in [('delta_c', 'delta'), ('length_c', 'length'),
                     ('enrol_week_c', 'enrol_week')]:
        df[col] = df[src] - df[src].mean()
    df['dummy_group'] = 1
    return df


def table(result, wanted_substrings):
    rows = []
    ci = result.conf_int()
    for term in result.params.index:
        if not any(w in term for w in wanted_substrings):
            continue
        rows.append({
            'term': term,
            'coef': float(result.params[term]),
            'z': float(result.tvalues[term]),
            'p': float(result.pvalues[term]),
            'ci_lo': float(ci.loc[term, 0]),
            'ci_hi': float(ci.loc[term, 1]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-rows', type=int, default=MAX_ROWS_FOR_MODEL)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='mechanisms to leave out, for control runs')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ITAD carries the label 'not_applicable' because clipping cannot affect
    # a percentile-based score; it belongs with the primary clipped condition
    df = prepare(MERGED, args.k, ['clipped', 'not_applicable'],
                 args.max_rows, args.seed)
    if args.exclude:
        df = df[~df['mechanism'].isin(args.exclude)]
        print('excluded %s' % args.exclude)
    print('rows used: %d' % len(df))
    print('per mechanism: %s' % df['mechanism'].value_counts().to_dict())
    print()

    fits = {}
    for name, formula in SPECS.items():
        result, method = _fit_with_fallback(formula, df, VC)
        fits[name] = result
        print('  %-12s llf=%14.2f  k=%3d  (%s)'
              % (name, result.llf, len(result.params), method))
    print()

    stat = 2 * (fits['interaction'].llf - fits['main'].llf)
    ddf = len(fits['interaction'].params) - len(fits['main'].params)
    print('delta by mechanism interaction: LR=%.2f df=%d p=%.3g'
          % (stat, ddf, sstats.chi2.sf(stat, ddf)))
    print()

    print('=' * 76)
    print('MAIN-EFFECTS MODEL, FIVE MECHANISMS')
    print('=' * 76)
    t = table(fits['main'], ['delta_c', 'mechanism', 'length_c',
                             'hardware_change', 'substitution'])
    print(t.to_string(index=False, float_format=lambda v: '%.6g' % v))
    t.to_csv(os.path.join(OUT_DIR, 'model_main_with_itad.csv'), index=False)
    print()

    print('=' * 76)
    print('INTERACTION TERMS')
    print('=' * 76)
    t2 = table(fits['interaction'], ['delta_c:'])
    print(t2.to_string(index=False, float_format=lambda v: '%.6g' % v))
    t2.to_csv(os.path.join(OUT_DIR, 'model_interaction_with_itad.csv'), index=False)
    print()

    print('=' * 76)
    print('ADDITIONAL COVARIATES')
    print('=' * 76)
    t3 = table(fits['extended'], ['enrol_week_c', 'delta_c:length_c'])
    print(t3.to_string(index=False, float_format=lambda v: '%.6g' % v))
    t3.to_csv(os.path.join(OUT_DIR, 'model_extended_with_itad.csv'), index=False)

    vc = fits['main'].vcomp
    print()
    print('variance components (participant, password): %s'
          % np.round(vc, 5).tolist())


if __name__ == '__main__':
    main()
