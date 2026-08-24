import os
RESULT_DIR = os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
"""
Reviewer-requested sensitivity check: does the central Delta x Mechanism
interaction finding survive under a genuine binomial (logistic) mixed
model, rather than the linear probability model used as the paper's
primary specification (Section 3.7)?

Fits statsmodels' BinomialBayesMixedGLM (variational Bayes) with the
identical fixed-effect formula and participant/password crossed random
effects as the primary linear model, on the identical 300,000-row
stratified subsample (same seed) used for the primary fit, so that any
difference in conclusions is attributable to the link function / outcome
distribution, not to a different sample of trials.

BinomialBayesMixedGLM is an approximate (variational Bayes) fit, not an
exact maximum-likelihood GLMM (no `lme4`-equivalent with crossed random
effects is available in statsmodels), and its ELBO is not a real
log-likelihood, so a likelihood-ratio joint test analogous to the linear
model's is not meaningful here. Instead we compare, term by term, whether
each Delta x Mechanism interaction coefficient's approximate posterior
95% credible interval excludes zero and has the same sign as the linear
model's estimate -- this directly answers "does the interaction survive
under a binomial outcome model," even without a joint test.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_model import subsample, MAX_ROWS_FOR_MODEL


def prep(df, seed):
    df = subsample(df, max_rows=MAX_ROWS_FOR_MODEL, seed=seed).copy()
    for col in ['participant_id', 'password_id', 'fold', 'mechanism', 'substitution_category']:
        df[col] = df[col].astype(str).astype(object)
    for col in ['delta', 'length', 'hardware_change', 'is_error']:
        df[col] = df[col].astype(float)
    df['delta_c'] = df['delta'] - df['delta'].mean()
    df['length_c'] = df['length'] - df['length'].mean()
    return df


def fit_glmm(df):
    fixed_formula = "is_error ~ delta_c * mechanism + length_c + C(substitution_category) + hardware_change + C(fold)"
    vc_formulas = {
        'participant': '0 + C(participant_id)',
        'password': '0 + C(password_id)',
    }
    model = BinomialBayesMixedGLM.from_formula(fixed_formula, vc_formulas, data=df)
    result = model.fit_vb()
    names = model.exog_names
    means = result.fe_mean
    sds = result.fe_sd
    from scipy.stats import norm
    rows = []
    for name, mean, sd in zip(names, means, sds):
        z = mean / sd if sd > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        rows.append({'term': name, 'post_mean': mean, 'post_sd': sd, 'z': z, 'p_approx': p})
    return pd.DataFrame(rows)


def run_one(label, path, seed):
    print(f'\n=== {label} (seed={seed}) ===')
    df = pd.read_parquet(path)
    df = prep(df, seed)
    print(f'n_rows = {len(df)}')
    result_df = fit_glmm(df)
    interaction_rows = result_df[result_df['term'].str.contains('delta_c:mechanism', na=False)]
    print(interaction_rows.to_string(index=False))
    return result_df


if __name__ == '__main__':
    runs = [
        ('primary (seed 0)', os.path.join(RESULT_DIR, 'mixed_model_input.parquet'), 0),
        ('replicate seed10', os.path.join(RESULT_DIR, 'replicate_seed10', 'mixed_model_input.parquet'), 0),
        ('replicate seed20', os.path.join(RESULT_DIR, 'replicate_seed20', 'mixed_model_input.parquet'), 0),
    ]
    all_results = {}
    for label, path, seed in runs:
        all_results[label] = run_one(label, path, seed)

    out_dir = os.path.join(os.environ.get('KD_RESULT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), 'stats_model')
    for label, df in all_results.items():
        safe = label.replace(' ', '_').replace('(', '').replace(')', '')
        df.to_csv(f'{out_dir}/glmm_sensitivity_{safe}.csv', index=False)
    print('\nDone. CSVs written to', out_dir)
