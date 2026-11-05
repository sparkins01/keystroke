"""
Revision experiments for TBIOM reviewer 2, points 2, 6 and 8, and the
associate editor's comment on password memorisation.

Both questions are extensions of the mixed-effects model of Section 3.7, and
both are answerable from the trial data already collected.

E2, reviewer point 8. Does password length change the *rate* of ageing, not
just the baseline error? The published model carries length as a main effect
only. Adding delta_c:length_c tests the interaction the reviewer asks for.

E3, reviewer points 2 and 6. Is the degradation template ageing, or users
still learning an unfamiliar password? The trial grid is two-dimensional on
(enrolment week i, query week j), so enrolment week and the gap Delta = j - i
are separately identified. A user who enrols in a later week has had more
practice, so a negative enrolment-week coefficient is a familiarisation
effect, and the Delta coefficient that survives beside it is ageing proper.

Note that only two of the three terms i, j and Delta can enter the model at
once, since j = i + Delta. Enrolment week and Delta are the pair we want.

Four specifications are fitted and compared by likelihood-ratio test:
  S0  the published model
  S1  S0 plus delta_c:length_c                      (E2)
  S2  S0 plus enrol_week_c                          (E3)
  S3  S0 plus both, with enrol_week_c:delta_c       (E2 and E3 together)

Each is also refitted as a logistic regression with participant-clustered
standard errors, because Section 4.3 showed that the choice of link function
changed two of three mechanism conclusions in the published analysis. A new
term is only reported as supported if it holds under both.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL, _fit_with_fallback

BASE = ("is_error ~ delta_c * mechanism + length_c "
        "+ C(substitution_category) + hardware_change + C(fold)")

SPECS = {
    'S0_published': BASE,
    'S1_length_x_delta': BASE + " + delta_c:length_c",
    'S2_enrol_week': BASE + " + enrol_week_c",
    'S3_both': BASE + " + delta_c:length_c + enrol_week_c + enrol_week_c:delta_c",
}

VC = {'participant': '0 + C(participant_id)', 'password': '0 + C(password_id)'}

RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))

REPLICATES = {
    'primary': os.path.join(RESULT_DIR, 'mixed_model_input.parquet'),
    'seed10': os.path.join(RESULT_DIR, 'replicate_seed10',
                           'mixed_model_input.parquet'),
    'seed20': os.path.join(RESULT_DIR, 'replicate_seed20',
                           'mixed_model_input.parquet'),
}


def prepare(path, max_rows, seed):
    df = pd.read_parquet(path)
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
    return df


def identification_report(df):
    """Enrolment week and Delta are negatively coupled by the triangular trial
    grid, so the correlation is worth stating rather than assuming away."""
    r = np.corrcoef(df['enrol_week'], df['delta'])[0, 1]
    grid = df.groupby(['enrol_week', 'delta']).size().unstack(fill_value=0)
    print("correlation(enrol_week, delta) = %+.3f" % r)
    print("non-empty (enrol_week, delta) cells: %d of %d"
          % (int((grid > 0).sum().sum()), grid.size))
    print()
    print("trials per cell (rows = enrolment week, columns = delta):")
    print(grid.to_string())
    print()
    return r


def fit_all(df, specs):
    fits = {}
    for name, formula in specs.items():
        t0 = time.time()
        result, method = _fit_with_fallback(formula, df, VC)
        fits[name] = result
        print("  %-20s llf=%14.2f  k=%3d  %5.0fs  (%s)"
              % (name, result.llf, len(result.params), time.time() - t0, method))
    return fits


def lr_test(full, reduced):
    stat = 2 * (full.llf - reduced.llf)
    ddf = len(full.params) - len(reduced.params)
    p = sstats.chi2.sf(stat, ddf) if ddf > 0 else np.nan
    return {'lr_stat': float(stat), 'df': int(ddf), 'p_value': float(p)}


def terms_of_interest(result, names):
    rows = []
    for term in names:
        if term not in result.params.index:
            continue
        rows.append({
            'term': term,
            'coef': float(result.params[term]),
            'se': float(result.bse[term]),
            'z': float(result.tvalues[term]),
            'p': float(result.pvalues[term]),
            'ci_lo': float(result.conf_int().loc[term, 0]),
            'ci_hi': float(result.conf_int().loc[term, 1]),
        })
    return pd.DataFrame(rows)


def logistic_check(df, specs):
    """Logistic refit with participant-clustered standard errors, mirroring
    glmm_logit_clusterrobust.py. No random effects: the clustering absorbs the
    within-participant correlation."""
    out = {}
    for name, formula in specs.items():
        model = smf.logit(formula, data=df)
        result = model.fit(disp=0, maxiter=200,
                           cov_type='cluster',
                           cov_kwds={'groups': df['participant_id']})
        out[name] = result
        print("  %-20s llf=%14.2f  converged=%s" % (name, result.llf, result.mle_retvals['converged']))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', default='primary', choices=list(REPLICATES))
    parser.add_argument('--max-rows', type=int, default=MAX_ROWS_FOR_MODEL)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-logit', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'revision')
    os.makedirs(out_dir, exist_ok=True)

    print("replicate: %s" % args.replicate)
    df = prepare(REPLICATES[args.replicate], args.max_rows, args.seed)
    print("rows used: %d" % len(df))
    print()
    print("=" * 72)
    print("IDENTIFICATION OF ENROLMENT WEEK AGAINST DELTA")
    print("=" * 72)
    corr = identification_report(df)

    print("=" * 72)
    print("LINEAR MIXED MODEL FITS")
    print("=" * 72)
    fits = fit_all(df, SPECS)
    print()

    print("=" * 72)
    print("NESTED MODEL COMPARISONS")
    print("=" * 72)
    comparisons = {
        'E2 length x delta (S1 vs S0)': lr_test(fits['S1_length_x_delta'], fits['S0_published']),
        'E3 enrolment week (S2 vs S0)': lr_test(fits['S2_enrol_week'], fits['S0_published']),
        'both (S3 vs S0)': lr_test(fits['S3_both'], fits['S0_published']),
    }
    for label, res in comparisons.items():
        print("  %-32s LR=%9.2f  df=%d  p=%.3g" % (label, res['lr_stat'], res['df'], res['p_value']))
    print()

    want = ['delta_c', 'length_c', 'delta_c:length_c', 'enrol_week_c',
            'enrol_week_c:delta_c']
    print("=" * 72)
    print("TERMS OF INTEREST (linear probability scale)")
    print("=" * 72)
    frames = []
    for name in ['S1_length_x_delta', 'S2_enrol_week', 'S3_both']:
        tab = terms_of_interest(fits[name], want)
        tab.insert(0, 'spec', name)
        frames.append(tab)
        print()
        print("--- %s ---" % name)
        print(tab.to_string(index=False, float_format=lambda v: '%.6g' % v))
    linear_terms = pd.concat(frames, ignore_index=True)
    linear_terms['link'] = 'linear'

    all_terms = [linear_terms]
    if not args.skip_logit:
        print()
        print("=" * 72)
        print("LOGISTIC REFIT, PARTICIPANT-CLUSTERED STANDARD ERRORS")
        print("=" * 72)
        logits = logistic_check(df, SPECS)
        frames = []
        for name in ['S1_length_x_delta', 'S2_enrol_week', 'S3_both']:
            tab = terms_of_interest(logits[name], want)
            tab.insert(0, 'spec', name)
            frames.append(tab)
            print()
            print("--- %s ---" % name)
            print(tab.to_string(index=False, float_format=lambda v: '%.6g' % v))
        logit_terms = pd.concat(frames, ignore_index=True)
        logit_terms['link'] = 'logit'
        all_terms.append(logit_terms)

    terms = pd.concat(all_terms, ignore_index=True)
    terms['replicate'] = args.replicate
    terms_path = os.path.join(out_dir, 'model_extensions_terms_%s.csv' % args.replicate)
    terms.to_csv(terms_path, index=False)

    comp = pd.DataFrame(comparisons).T.reset_index().rename(columns={'index': 'comparison'})
    comp['replicate'] = args.replicate
    comp['corr_enrol_delta'] = corr
    comp_path = os.path.join(out_dir, 'model_extensions_lrtests_%s.csv' % args.replicate)
    comp.to_csv(comp_path, index=False)

    print()
    print("wrote %s" % terms_path)
    print("wrote %s" % comp_path)


if __name__ == '__main__':
    main()
